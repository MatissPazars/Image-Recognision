class_name DrawCanvas
extends Control
## A canvas the user can draw on with the mouse (left = draw, right = erase).
## `get_mnist_pixels()` converts the drawing into a 28x28 MNIST-style input:
## the digit is cropped, scaled to fit a 20x20 box and centred by its
## centre of mass inside a 28x28 image, exactly like the MNIST dataset.

signal drawing_changed

const SIZE := 280
@export var brush_radius := 11.0

var _data := PackedByteArray()
var _image: Image
var _texture: ImageTexture
var _last_pos := Vector2.ZERO
var _drawing := false
var _erasing := false


func _ready() -> void:
    custom_minimum_size = custom_minimum_size.max(Vector2(SIZE, SIZE))
    size_flags_horizontal = Control.SIZE_SHRINK_BEGIN
    mouse_filter = Control.MOUSE_FILTER_STOP
    _data.resize(SIZE * SIZE)
    _image = Image.create_from_data(SIZE, SIZE, false, Image.FORMAT_L8, _data)
    _texture = ImageTexture.create_from_image(_image)


func clear() -> void:
    _data.fill(0)
    _refresh()
    drawing_changed.emit()


func is_empty() -> bool:
    for v in _data:
        if v != 0:
            return false
    return true


func _draw() -> void:
    draw_texture_rect(_texture, Rect2(Vector2.ZERO, size), false)
    draw_rect(Rect2(Vector2.ZERO, size), Color(0.4, 0.6, 1.0), false, 2.0)


func _gui_input(event: InputEvent) -> void:
    if event is InputEventMouseButton:
        var mb := event as InputEventMouseButton
        if mb.button_index == MOUSE_BUTTON_LEFT or mb.button_index == MOUSE_BUTTON_RIGHT:
            _drawing = mb.pressed
            _erasing = mb.button_index == MOUSE_BUTTON_RIGHT
            if mb.pressed:
                _last_pos = _to_canvas(mb.position)
                _stamp_line(_last_pos, _last_pos)
            accept_event()
    elif event is InputEventMouseMotion and _drawing:
        var p := _to_canvas((event as InputEventMouseMotion).position)
        _stamp_line(_last_pos, p)
        _last_pos = p
        accept_event()


func _to_canvas(p: Vector2) -> Vector2:
    return p * Vector2(SIZE, SIZE) / size


func _stamp_line(a: Vector2, b: Vector2) -> void:
    var dist := a.distance_to(b)
    var steps := maxi(1, int(dist / (brush_radius * 0.3)))
    for i in steps + 1:
        _stamp(a.lerp(b, float(i) / steps))
    _refresh()
    drawing_changed.emit()


func _stamp(c: Vector2) -> void:
    var r := brush_radius
    var x0 := maxi(0, int(c.x - r - 1))
    var x1 := mini(SIZE - 1, int(c.x + r + 1))
    var y0 := maxi(0, int(c.y - r - 1))
    var y1 := mini(SIZE - 1, int(c.y + r + 1))
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            var d := Vector2(x + 0.5, y + 0.5).distance_to(c)
            if d > r:
                continue
            var i := y * SIZE + x
            if _erasing:
                _data[i] = 0
            else:
                # soft edge for anti-aliasing (like MNIST's blurry strokes)
                var v := int(clampf((r - d) / 2.5, 0.0, 1.0) * 255.0)
                if v > _data[i]:
                    _data[i] = v


func _refresh() -> void:
    _image.set_data(SIZE, SIZE, false, Image.FORMAT_L8, _data)
    _texture.update(_image)
    queue_redraw()


## Returns 784 floats (0..1) or an empty array if nothing is drawn.
func get_mnist_pixels() -> PackedFloat32Array:
    # 1. bounding box of the drawing
    var min_x := SIZE; var min_y := SIZE; var max_x := -1; var max_y := -1
    for y in SIZE:
        var row := y * SIZE
        for x in SIZE:
            if _data[row + x] > 0:
                min_x = mini(min_x, x); max_x = maxi(max_x, x)
                min_y = mini(min_y, y); max_y = maxi(max_y, y)
    if max_x < 0:
        return PackedFloat32Array()
    var w := max_x - min_x + 1
    var h := max_y - min_y + 1

    # 2. crop and scale so the longest side is 20 px (keeps aspect ratio)
    var cropped := _image.get_region(Rect2i(min_x, min_y, w, h))
    var factor := 20.0 / maxf(w, h)
    var nw := clampi(roundi(w * factor), 1, 20)
    var nh := clampi(roundi(h * factor), 1, 20)
    # downscale in two steps for smoother results
    if w > 80 or h > 80:
        cropped.resize(maxi(nw * 4, 1), maxi(nh * 4, 1), Image.INTERPOLATE_BILINEAR)
    cropped.resize(nw, nh, Image.INTERPOLATE_BILINEAR)
    var small := cropped.get_data()

    # 3. centre of mass of the scaled digit
    var sum := 0.0; var cx := 0.0; var cy := 0.0
    for y in nh:
        for x in nw:
            var v := float(small[y * nw + x])
            sum += v; cx += v * x; cy += v * y
    if sum <= 0.0:
        return PackedFloat32Array()
    cx /= sum; cy /= sum

    # 4. paste into 28x28 so the centre of mass is at (14, 14)
    var ox := roundi(14.0 - cx - 0.5)
    var oy := roundi(14.0 - cy - 0.5)
    ox = clampi(ox, 0, 28 - nw)
    oy = clampi(oy, 0, 28 - nh)
    var out := PackedFloat32Array(); out.resize(784)
    for y in nh:
        for x in nw:
            out[(y + oy) * 28 + (x + ox)] = small[y * nw + x] / 255.0
    return out
