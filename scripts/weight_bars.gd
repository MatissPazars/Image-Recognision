class_name WeightBars
extends Control
## Draws one horizontal bar per digit (0-9), growing right for positive
## values (green) and left for negative values (red). Clicking a row emits
## `bar_clicked(index)`.

signal bar_clicked(index: int)

var values := PackedFloat32Array()
var highlight := -1
var label_width := 28.0


func set_values(v: PackedFloat32Array, highlight_index: int = -1) -> void:
    values = v
    highlight = highlight_index
    queue_redraw()


func _draw() -> void:
    var n := values.size()
    if n == 0:
        return
    var font := get_theme_default_font()
    var fs := 14
    var row_h := size.y / n
    var bar_area := size.x - label_width - 56.0
    var center := label_width + bar_area * 0.5
    var max_abs := 0.0001
    for v in values:
        max_abs = maxf(max_abs, absf(v))
    draw_line(Vector2(center, 0), Vector2(center, size.y), Color(1, 1, 1, 0.25), 1.0)
    for i in n:
        var y := i * row_h
        var v := values[i]
        var w := v / max_abs * bar_area * 0.5
        var col := Color(0.35, 0.9, 0.45) if v >= 0.0 else Color(0.95, 0.35, 0.35)
        if i == highlight:
            draw_rect(Rect2(0, y, size.x, row_h), Color(1, 1, 1, 0.08))
        var r := Rect2(center, y + row_h * 0.2, w, row_h * 0.6).abs()
        draw_rect(r, col)
        draw_string(font, Vector2(6, y + row_h * 0.5 + fs * 0.35), str(i),
                HORIZONTAL_ALIGNMENT_LEFT, -1, fs, Color.WHITE)
        draw_string(font, Vector2(size.x - 52, y + row_h * 0.5 + fs * 0.35), "%+.2f" % v,
                HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.8, 0.8, 0.8))


func _gui_input(event: InputEvent) -> void:
    if event is InputEventMouseButton and event.pressed and event.button_index == MOUSE_BUTTON_LEFT:
        if values.size() > 0:
            var i := clampi(int(event.position.y / (size.y / values.size())), 0, values.size() - 1)
            bar_clicked.emit(i)
            accept_event()
