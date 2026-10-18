class_name NeuronThumb
extends Button
## A clickable thumbnail showing one hidden neuron's 28x28 weight pattern,
## its index, a value (e.g. activation) and an activation bar.

const THUMB := 64

var index := 0
var _tex_rect: TextureRect
var _label: Label
var _bar: ColorRect


func _init() -> void:
    toggle_mode = true
    focus_mode = Control.FOCUS_NONE
    custom_minimum_size = Vector2(THUMB + 12, THUMB + 32)

    var pressed_style := StyleBoxFlat.new()
    pressed_style.bg_color = Color(0.25, 0.2, 0.1)
    pressed_style.set_border_width_all(3)
    pressed_style.border_color = Color(1.0, 0.65, 0.2)
    pressed_style.set_corner_radius_all(4)
    add_theme_stylebox_override("pressed", pressed_style)
    add_theme_stylebox_override("hover_pressed", pressed_style)

    _tex_rect = TextureRect.new()
    _tex_rect.position = Vector2(6, 6)
    _tex_rect.size = Vector2(THUMB, THUMB)
    _tex_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
    _tex_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
    _tex_rect.mouse_filter = Control.MOUSE_FILTER_IGNORE
    add_child(_tex_rect)

    _label = Label.new()
    _label.position = Vector2(2, THUMB + 7)
    _label.size = Vector2(THUMB + 8, 16)
    _label.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    _label.add_theme_font_size_override("font_size", 11)
    _label.mouse_filter = Control.MOUSE_FILTER_IGNORE
    add_child(_label)

    _bar = ColorRect.new()
    _bar.position = Vector2(6, THUMB + 25)
    _bar.size = Vector2(0, 4)
    _bar.color = Color(1.0, 0.65, 0.2)
    _bar.mouse_filter = Control.MOUSE_FILTER_IGNORE
    add_child(_bar)


func setup(i: int, texture: Texture2D) -> void:
    index = i
    _tex_rect.texture = texture
    tooltip_text = "Neuron #%d" % i
    set_info("#%d" % i, 0.0)


## text: caption under the thumbnail; ratio: 0..1 fill of the activation bar.
func set_info(text: String, ratio: float) -> void:
    _label.text = text
    _bar.size.x = THUMB * clampf(ratio, 0.0, 1.0)
