import argparse
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from svgpathtools import parse_path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from core.preprocessing import load_image, save_png, grayscale, prepare_gray

CLASSES = {"window": 0, "door": 1, "sink": 2, "toilet": 3}
COLORS = {"window": (220, 60, 240), "door": (255, 180, 50),
          "sink": (70, 200, 100), "toilet": (255, 100, 100)}
NUMBERS = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


def nums(value):
    return [float(m) for m in NUMBERS.findall(value or "")]


def tag(element):
    return element.tag.rsplit("}", 1)[-1]


def translate(x, y):
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=float)


def svg_transform(value):
    result = np.eye(3, dtype=float)
    for name, data in re.findall(r"([A-Za-z]+)\s*\(([^)]*)\)", value or ""):
        v = nums(data)
        m = np.eye(3, dtype=float)
        name = name.lower()
        if name == "matrix" and len(v) == 6:
            a, b, c, d, e, f = v
            m = np.array([[a, c, e], [b, d, f], [0, 0, 1]], dtype=float)
        elif name == "translate" and v:
            m = translate(v[0], v[1] if len(v) > 1 else 0)
        elif name == "scale" and v:
            m[0, 0] = v[0]
            m[1, 1] = v[1] if len(v) > 1 else v[0]
        elif name == "rotate" and v:
            t = math.radians(v[0])
            m[:2, :2] = [[math.cos(t), -math.sin(t)],
                          [math.sin(t), math.cos(t)]]
            if len(v) > 2:
                m = translate(v[1], v[2]) @ m @ translate(-v[1], -v[2])
        elif name == "skewx" and v:
            m[0, 1] = math.tan(math.radians(v[0]))
        elif name == "skewy" and v:
            m[1, 0] = math.tan(math.radians(v[0]))
        else:
            raise ValueError(f"Unsupported SVG transform: {name}")
        result = result @ m
    return result


def primitive_points(element):
    kind = tag(element)
    if kind in ("polygon", "polyline"):
        data = nums(element.get("points", ""))
        if len(data) < 4 or len(data) % 2:
            return []
        return list(zip(data[::2], data[1::2]))
    if kind == "rect":
        x = float(element.get("x", 0))
        y = float(element.get("y", 0))
        w = float(element.get("width", 0))
        h = float(element.get("height", 0))
        return [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
    if kind == "path":
        output = []
        try:
            path = parse_path(element.get("d", ""))
            for segment in path:
                for t in np.linspace(0.0, 1.0, 21):
                    point = segment.point(float(t))
                    output.append((point.real, point.imag))
        except (ValueError, TypeError, ZeroDivisionError):
            return []
        return output
    return []


def transform_points(points, matrix):
    if not points:
        return []
    xy = np.asarray(points, dtype=float)
    xy1 = np.column_stack((xy, np.ones(len(xy))))
    return (matrix @ xy1.T).T[:, :2].tolist()


def collect_shapes(element, parent_matrix, allowed):
    matrix = parent_matrix @ svg_transform(element.get("transform", ""))
    points = []
    if tag(element) in allowed:
        points.extend(transform_points(primitive_points(element), matrix))
    for child in element:
        points.extend(collect_shapes(child, matrix, allowed))
    return points


def object_label(element):
    if tag(element) != "g":
        return None
    if element.get("id") == "Window":
        return "window"
    if element.get("id") == "Door":
        return "door"
    classes = set(element.get("class", "").split())
    if "FixedFurniture" in classes:
        if "Sink" in classes:
            return "sink"
        if "Toilet" in classes:
            return "toilet"
    return None


def object_geometry(element, matrix, label):
    points = []
    if label in ("window", "door"):
        for child in element:
            if tag(child) in ("polygon", "rect"):
                points.extend(collect_shapes(child, matrix, {"polygon", "rect"}))
        if label == "door":
            for child in element:
                points.extend(collect_shapes(child, matrix, {"path"}))
    else:
        for child in element:
            classes = set(child.get("class", "").split())
            if "BoundaryPolygon" in classes:
                points.extend(collect_shapes(child, matrix, {"polygon", "rect"}))
        if not points:
            for child in element:
                points.extend(collect_shapes(child, matrix, {"polygon", "rect"}))
    return points


def read_svg(svg_path):
    root = ET.parse(svg_path).getroot()
    viewbox = nums(root.get("viewBox", ""))
    if len(viewbox) != 4 or min(viewbox[2:]) <= 0:
        raise ValueError("Invalid SVG viewBox")
    objects, walls = [], []

    def visit(element, parent_matrix):
        matrix = parent_matrix @ svg_transform(element.get("transform", ""))
        label = object_label(element)
        if label:
            points = object_geometry(element, matrix, label)
            if points:
                objects.append({
                    "class": label,
                    "class_id": CLASSES[label],
                    "points": np.asarray(points, dtype=float)
                })
        if tag(element) == "g" and element.get("id") == "Wall":
            for child in element:
                if tag(child) in ("polygon", "rect"):
                    points = collect_shapes(child, matrix, {"polygon", "rect"})
                    if len(points) >= 3:
                        walls.append(np.asarray(points, dtype=float))
        for child in element:
            visit(child, matrix)

    visit(root, np.eye(3, dtype=float))
    return viewbox, objects, walls


def rotated_svg_points(points, viewbox, angle):
    vx, vy, vw, vh = viewbox
    x, y = points[:, 0]-vx, points[:, 1]-vy
    if angle == 0:
        return np.column_stack([x,y]), vw, vh
    if angle == 90:
        return np.column_stack([vh-y,x]), vh, vw
    if angle == 180:
        return np.column_stack([vw-x,vh-y]), vw, vh
    if angle == 270:
        return np.column_stack([y,vw-x]), vh, vw
    raise ValueError("Invalid orientation")


def project(points, viewbox, target_shape, params):
    angle, factor, dx, dy = params
    h, w = target_shape[:2]
    rotated, svg_w, svg_h = rotated_svg_points(points, viewbox, angle)
    scale = min(w/svg_w, h/svg_h) * factor
    ox = (w-svg_w*scale)/2 + dx*w
    oy = (h-svg_h*scale)/2 + dy*h
    return rotated*scale + [ox, oy]


def sample_wall_edges(walls, step=14.0, maximum=2200):
    samples = []
    for polygon in walls:
        ring = np.vstack((polygon, polygon[0]))
        for a, b in zip(ring[:-1], ring[1:]):
            n = max(2, int(np.linalg.norm(b-a)/step)+1)
            t = np.linspace(0, 1, n)[:, None]
            samples.append(a[None,:]*(1-t) + b[None,:]*t)
    if not samples:
        raise ValueError("SVG has no usable wall polygons")
    xy = np.vstack(samples)
    if len(xy) > maximum:
        xy = xy[np.linspace(0,len(xy)-1,maximum).astype(int)]
    return xy


def align_svg(image, viewbox, walls, orientation="auto"):
    """
    Approximate grayscale edge registration.
    Returns params and an error in thumbnail pixels.
    Manual review remains mandatory before exporting training crops.
    """
    if not walls:
        raise ValueError("No walls for SVG alignment")
    gray = grayscale(image)
    h, w = gray.shape
    ratio = min(1.0, 600.0/max(h,w))
    thumb = cv2.resize(gray, (max(1,round(w*ratio)),
                              max(1,round(h*ratio))),
                       interpolation=cv2.INTER_AREA)
    smoothed = cv2.GaussianBlur(thumb, (3,3), 0)
    edges = cv2.Canny(smoothed, 30, 110)
    distance = cv2.distanceTransform(255-edges,cv2.DIST_L2,3)
    th, tw = thumb.shape
    samples = sample_wall_edges(walls)
    angles = (0, 90, 180, 270) if orientation == "auto" else (int(orientation),)

    def score(params):
        xy = project(samples,viewbox,thumb.shape,params)
        x = np.rint(xy[:,0]).astype(int)
        y = np.rint(xy[:,1]).astype(int)
        inside = (x>=0)&(x<tw)&(y>=0)&(y<th)
        if inside.mean() < .60:
            return 1000.0
        values = distance[y[inside],x[inside]]
        return float(np.minimum(values,15).mean()
                     + 18*(1-inside.mean())
                     + .4*abs(params[1]-1))

    best, minimum = None, float("inf")
    for angle in angles:
        for factor in (.82,.90,.97,1.0,1.04,1.10,1.18):
            for dx in (-.12,-.06,0,.06,.12):
                for dy in (-.12,-.06,0,.06,.12):
                    params = (angle,factor,dx,dy)
                    value = score(params)
                    if value < minimum:
                        best, minimum = params, value
    angle, factor, dx, dy = best
    for f in np.linspace(factor-.04,factor+.04,5):
        for x in np.linspace(dx-.03,dx+.03,5):
            for y in np.linspace(dy-.03,dy+.03,5):
                params = (angle,float(f),float(x),float(y))
                value = score(params)
                if value < minimum:
                    best, minimum = params, value
    return best, minimum, thumb.shape


def transform_to_image(points, viewbox, image_shape, thumb_shape, params):
    xy = project(points,viewbox,thumb_shape,params)
    xy[:,0] *= image_shape[1]/thumb_shape[1]
    xy[:,1] *= image_shape[0]/thumb_shape[0]
    return xy


def extract_annotations(objects, viewbox, image_shape, thumb_shape, params):
    h, w = image_shape[:2]
    result = []
    for i, obj in enumerate(objects):
        xy = transform_to_image(obj["points"],viewbox,image_shape,thumb_shape,params)
        xy_min, xy_max = xy.min(axis=0), xy.max(axis=0)
        x1,y1 = map(float,xy_min)
        x2,y2 = map(float,xy_max)
        if x2<=0 or y2<=0 or x1>=w or y1>=h:
            continue
        x1,x2 = max(0,x1),min(w,x2)
        y1,y2 = max(0,y1),min(h,y2)
        if x2-x1 < 2 or y2-y1 < 2:
            continue
        result.append({
            "id": i,
            "class": obj["class"],
            "class_id": obj["class_id"],
            "bbox": [round(x1,2),round(y1,2),round(x2,2),round(y2,2)]
        })
    return result


def make_preview(image, annotations):
    preview = image.copy()
    for obj in annotations:
        x1,y1,x2,y2 = map(lambda v: int(round(v)),obj["bbox"])
        color = COLORS[obj["class"]]
        cv2.rectangle(preview,(x1,y1),(x2,y2),color,2)
        cv2.putText(preview,f'{obj["id"]} {obj["class"]}',
                    (x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,2)
    return preview


def wall_preview(image,walls,viewbox,thumb_shape,params):
    preview = image.copy()
    for polygon in walls:
        xy = transform_to_image(polygon,viewbox,image.shape,thumb_shape,params)
        cv2.polylines(preview,[np.rint(xy).astype(np.int32)],
                      True,(255,0,255),2)
    return preview


def save_yolo(path,annotations,width,height):
    rows=[]
    for obj in annotations:
        x1,y1,x2,y2=obj["bbox"]
        rows.append(
            f'{obj["class_id"]} {(x1+x2)/2/width:.6f} '
            f'{(y1+y2)/2/height:.6f} {(x2-x1)/width:.6f} '
            f'{(y2-y1)/height:.6f}'
        )
    Path(path).write_text("\n".join(rows)+("\n" if rows else ""),encoding="utf-8")


def overlap_ratio(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1])
    x2,y2=min(a[2],b[2]),min(a[3],b[3])
    overlap=max(0,x2-x1)*max(0,y2-y1)
    area_a=max(1,(a[2]-a[0])*(a[3]-a[1]))
    return overlap/area_a


def padded_box(box,image_shape,pad_fraction=.15):
    h,w=image_shape[:2]
    x1,y1,x2,y2=box
    extra=max(4,int(max(x2-x1,y2-y1)*pad_fraction))
    return [max(0,int(x1)-extra),max(0,int(y1)-extra),
            min(w,int(math.ceil(x2))+extra),
            min(h,int(math.ceil(y2))+extra)]


def letterbox_crop(gray,box,size):
    x1,y1,x2,y2=map(int,box)
    crop=gray[y1:y2,x1:x2]
    if crop.size==0:
        raise ValueError("Empty crop")
    h,w=crop.shape
    factor=min(size/w,size/h)
    nw=max(1,min(size,int(round(w*factor))))
    nh=max(1,min(size,int(round(h*factor))))
    interpolation=cv2.INTER_AREA if factor<1 else cv2.INTER_LINEAR
    crop=cv2.resize(crop,(nw,nh),interpolation=interpolation)
    result=np.full((size,size),255,dtype=np.uint8)
    px=(size-nw)//2
    py=(size-nh)//2
    result[py:py+nh,px:px+nw]=crop
    return result


def export_symbol_samples(image,annotations,folder,size=96,
                          negative_ratio=1.0,seed=42):
    """
    Export positive symbol templates, hard-ish background crops and pairs.
    Keep train/val/test splits at PLAN level in the future dataset builder.
    """
    folder=Path(folder)
    rng=random.Random(seed)
    processed,_=prepare_gray(image,"auto")
    h,w=processed.shape
    plan_id=Path(folder).parent.name
    examples=[]
    positive_boxes=[]

    for i,obj in enumerate(annotations):
        bounds=padded_box(obj["bbox"],processed.shape)
        if bounds[2]-bounds[0]<3 or bounds[3]-bounds[1]<3:
            continue
        patch=letterbox_crop(processed,bounds,size)
        file=Path("samples")/obj["class"]/f"positive_{i:05d}.png"
        save_png(folder/file,patch)
        examples.append({
            "file":file.as_posix(),
            "kind":"positive",
            "class":obj["class"],
            "class_id":obj["class_id"],
            "bbox":obj["bbox"],
            "crop_bbox":bounds,
            "plan_id":plan_id
        })
        positive_boxes.append(bounds)

    positive_count=len(examples)
    negative_goal=int(math.ceil(positive_count*max(0,negative_ratio)))
    boxes_for_sizes=positive_boxes[:]
    attempts=0
    negative_count=0

    while negative_count<negative_goal and attempts<max(1500,negative_goal*90):
        attempts+=1
        if not boxes_for_sizes:
            break
        exemplar=rng.choice(boxes_for_sizes)
        box_w=max(12,exemplar[2]-exemplar[0])
        box_h=max(12,exemplar[3]-exemplar[1])
        if box_w>=w or box_h>=h:
            continue
        x=rng.randint(0,w-box_w)
        y=rng.randint(0,h-box_h)
        candidate=[x,y,x+box_w,y+box_h]
        if any(overlap_ratio(candidate,obj["bbox"])>.01 for obj in annotations):
            continue
        patch=processed[y:y+box_h,x:x+box_w]
        ink_fraction=float((patch<205).mean())
        if not .025<=ink_fraction<=.75:
            continue
        output=letterbox_crop(processed,candidate,size)
        file=Path("samples")/"background"/f"negative_{negative_count:05d}.png"
        save_png(folder/file,output)
        examples.append({
            "file":file.as_posix(),
            "kind":"negative",
            "class":"background",
            "class_id":-1,
            "bbox":candidate,
            "crop_bbox":candidate,
            "plan_id":plan_id
        })
        negative_count+=1

    (folder/"samples.jsonl").write_text(
        "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in examples),
        encoding="utf-8"
    )

    groups={}
    for e in examples:
        groups.setdefault(e["class"],[]).append(e["file"])
    pairs=[]
    for name,items in groups.items():
        if name=="background" or len(items)<2:
            continue
        for _ in range(min(48,len(items)*3)):
            a,b=rng.sample(items,2)
            pairs.append({"a":a,"b":b,"same":1,"class":name})
    wanted=len(pairs)
    candidates=list(examples)
    if len(candidates)>=2:
        tries=0
        negatives=0
        while negatives<wanted and tries<wanted*20+50:
            tries+=1
            a,b=rng.sample(candidates,2)
            if a["class"]==b["class"]:
                continue
            pairs.append({
                "a":a["file"],"b":b["file"],
                "same":0,"class_a":a["class"],"class_b":b["class"]
            })
            negatives+=1
    rng.shuffle(pairs)
    (folder/"pairs.jsonl").write_text(
        "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in pairs),
        encoding="utf-8"
    )
    (folder/"samples_manifest.json").write_text(
        json.dumps({
            "template_size":size,
            "positive_count":positive_count,
            "negative_count":negative_count,
            "pair_count":len(pairs),
            "classes":CLASSES,
            "split_policy":"split_by_plan_id_before_training"
        },indent=2),encoding="utf-8"
    )
    print(f"Samples: {positive_count} positive, {negative_count} negative")
    print(f"Pairs: {len(pairs)} | folder: {folder}")


def main():
    parser=argparse.ArgumentParser(
        description="Blueprint Whisperer - SVG symbol dataset for Siamese CNN")
    parser.add_argument("--image",required=True,type=Path)
    parser.add_argument("--svg",type=Path)
    parser.add_argument("--out",type=Path,default=Path("data/annotations/preview"))
    parser.add_argument("--rotation",choices=["auto","0","90","180","270"],
                        default="auto")
    parser.add_argument("--export-crops",action="store_true")
    parser.add_argument("--accept-alignment",action="store_true")
    parser.add_argument("--size",type=int,default=96)
    parser.add_argument("--negative-ratio",type=float,default=1.0)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--scale-adjust", type=float, default=1.0)
    parser.add_argument("--shift-x", type=float, default=0.0)
    parser.add_argument("--shift-y", type=float, default=0.0)
    args=parser.parse_args()

    image=load_image(args.image)
    svg_path=args.svg or args.image.with_name("model.svg")
    if not svg_path.is_file():
        parser.error(f"SVG not found: {svg_path}")
    args.out.mkdir(parents=True,exist_ok=True)

    viewbox,objects,walls=read_svg(svg_path)
    params,error,thumb_shape=align_svg(
        image,viewbox,walls,args.rotation
    )

    angle, scale, dx, dy = params

    params = (
    angle,
    scale * args.scale_adjust,
    dx + args.shift_x,
    dy + args.shift_y
    )

    annotations=extract_annotations(
        objects,viewbox,image.shape,thumb_shape,params
    )
    save_png(args.out/"preview.png",make_preview(image,annotations))
    save_png(args.out/"wall_alignment.png",wall_preview(
        image,walls,viewbox,thumb_shape,params
    ))
    save_yolo(args.out/"labels.txt",annotations,image.shape[1],image.shape[0])
    (args.out/"annotations.json").write_text(
        json.dumps(annotations,indent=2),encoding="utf-8"
    )
    counts=Counter(x["class"] for x in annotations)
    alignment={
        "image":str(args.image.resolve()),
        "svg":str(svg_path.resolve()),
        "viewbox":viewbox,
        "image_dimensions":[image.shape[1],image.shape[0]],
        "rotation":params[0],
        "scale_multiplier":params[1],
        "normalized_shift":[params[2],params[3]],
        "edge_distance_score":round(error,4),
        "review_required":True,
        "object_count":dict(counts)
    }
    (args.out/"alignment.json").write_text(
        json.dumps(alignment,indent=2),encoding="utf-8"
    )

    if error>7:
        print("WARNING: High edge alignment error. Inspect previews.")
    print("Objects:",dict(counts))
    print("Review:",args.out/"wall_alignment.png")
    print("Preview:",args.out/"preview.png")

    if args.export_crops:
        if not args.accept_alignment:
            parser.error(
                "First review wall_alignment.png and preview.png. "
                "Then add --accept-alignment to export training crops."
            )
        if args.size<32:
            parser.error("--size must be at least 32")
        export_symbol_samples(
            image,annotations,args.out/"siamese",
            size=args.size,
            negative_ratio=args.negative_ratio,
            seed=args.seed
        )


if __name__=="__main__":
    main()
