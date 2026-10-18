class_name PatternViewer
extends HBoxContainer
## Interactive explorer for the hidden-layer "patterns" of the network.
##
## Every hidden neuron has 784 input weights - one per pixel - which can be
## shown as a 28x28 image: orange = "I want ink here", blue = "I want this
## pixel empty". The viewer lets you browse, sort and search all neurons,
## see how strongly each one reacts to your current drawing, and how each
## neuron votes for the 10 digits.

enum SortMode { INDEX, ACTIVATION, VOTE_FOR, VOTE_AGAINST, STRENGTH }
const SORT_NAMES := [
    "Neuron number",
    "Most active on your drawing",
    "Strongest vote FOR digit",
    "Strongest vote AGAINST digit",
    "Overall weight strength",
]

const ORANGE := Color(1.0, 0.6, 0.15)
const BLUE := Color(0.25, 0.55, 1.0)
const ZERO := Color(0.07, 0.07, 0.09)
const MASKED := Color(0.13, 0.13, 0.15)

var net: NeuralNetwork
## 784 bytes: 1 = pixel contains ink in MNIST, 0 = (almost) never used.
var pixel_mask := PackedByteArray()
var hide_unused := true
var _mask_thread: Thread
var pixels := PackedFloat32Array()
var acts := PackedFloat32Array()
var strengths := PackedFloat32Array()
var order := PackedInt32Array()
var selected := 0
var sort_mode := SortMode.INDEX
var sort_digit := 0

var _thumbs: Array[NeuronThumb] = []
var _textures: Array[ImageTexture] = []
var _template_textures: Array[ImageTexture] = []
var _group := ButtonGroup.new()
var _updating_ui := false

# UI
var _flow: HFlowContainer
var _scroll: ScrollContainer
var _sort_option: OptionButton
var _digit_spin: SpinBox
var _count_label: Label
var _title: Label
var _index_spin: SpinBox
var _big_weights: TextureRect
var _big_contrib: TextureRect
var _stats: Label
var _vote_bars: WeightBars
var _contrib_tex: ImageTexture
var _big_tex: ImageTexture


func _ready() -> void:
    _build_ui()
    # Pixels that (almost) never contain ink in MNIST never receive training
    # updates, so their weights stay random noise. Find them in the background
    # so they can be hidden from the pattern images.
    _mask_thread = Thread.new()
    _mask_thread.start(_compute_mask)


func _exit_tree() -> void:
    if _mask_thread and _mask_thread.is_started():
        _mask_thread.wait_to_finish()


# ------------------------------------------------------------------ public API

## Call whenever the network changes (loaded, trained, reset).
func set_network(n: NeuralNetwork) -> void:
    net = n
    var H := net.hidden
    if _thumbs.size() != H:
        _rebuild_thumbs(H)
    strengths.resize(H)
    for h in H:
        var w := net.neuron_weights(h)
        var s := 0.0
        for v in w:
            s += v * v
        strengths[h] = sqrt(s)
    _redraw_patterns()
    selected = clampi(selected, 0, H - 1)
    _index_spin.max_value = H - 1
    acts = net.hidden_activations(pixels)
    _apply_sort()
    _select(selected)


## Call whenever the drawing changes. `p` = 784 floats (or empty).
func set_input(p: PackedFloat32Array) -> void:
    pixels = p
    if net == null:
        return
    acts = net.hidden_activations(pixels)
    if sort_mode == SortMode.ACTIVATION:
        _apply_sort()
    else:
        _refresh_thumb_info()
    _update_detail()


# ------------------------------------------------------------------ rendering

func _compute_mask() -> void:
    var imgs := MnistLoader.load_images(MnistLoader.TRAIN_IMAGES)
    var n := imgs.size() / 784
    var counts := PackedInt32Array(); counts.resize(784)
    var sampled := 0
    for s in range(0, n, 4):   # every 4th image is plenty
        var off := s * 784
        sampled += 1
        for i in 784:
            if imgs[off + i] != 0:
                counts[i] += 1
    var m := PackedByteArray(); m.resize(784)
    var threshold := maxi(1, int(sampled * 0.002))  # inked in < 0.2% of digits
    for i in 784:
        m[i] = 1 if counts[i] >= threshold else 0
    _on_mask_ready.call_deferred(m)


func _on_mask_ready(m: PackedByteArray) -> void:
    _mask_thread.wait_to_finish()
    pixel_mask = m
    if net:
        _redraw_patterns()
        _update_detail()


func _on_hide_toggled(on: bool) -> void:
    hide_unused = on
    if net:
        _redraw_patterns()
        _update_detail()


func _redraw_patterns() -> void:
    for h in net.hidden:
        _textures[h].update(make_weight_image(net.neuron_weights(h)))
    _update_templates()


static func weight_color(v: float, m: float) -> Color:
    var t := clampf(v / m, -1.0, 1.0)
    return ZERO.lerp(ORANGE, t) if t >= 0.0 else ZERO.lerp(BLUE, -t)


## Turns 784 signed values into a 28x28 orange/blue image.
## Masked (never-used) pixels are drawn as plain dark background.
func make_weight_image(w: PackedFloat32Array) -> Image:
    var use_mask := hide_unused and pixel_mask.size() == 784
    var m := 0.0001
    for i in 784:
        if not use_mask or pixel_mask[i] == 1:
            m = maxf(m, absf(w[i]))
    m *= 0.8  # a bit of extra contrast
    var data := PackedByteArray(); data.resize(784 * 3)
    for i in 784:
        var c := MASKED if (use_mask and pixel_mask[i] == 0) else weight_color(w[i], m)
        data[i * 3] = c.r8
        data[i * 3 + 1] = c.g8
        data[i * 3 + 2] = c.b8
    return Image.create_from_data(28, 28, false, Image.FORMAT_RGB8, data)


func _rebuild_thumbs(H: int) -> void:
    for t in _thumbs:
        _flow.remove_child(t)
        t.queue_free()
    _thumbs.clear()
    _textures.clear()
    var blank := Image.create(28, 28, false, Image.FORMAT_RGB8)
    for h in H:
        var tex := ImageTexture.create_from_image(blank)
        _textures.append(tex)
        var t := NeuronThumb.new()
        t.button_group = _group
        t.setup(h, tex)
        t.pressed.connect(_select.bind(h))
        _flow.add_child(t)
        _thumbs.append(t)


## "Digit templates": for every output digit, sum all neuron patterns
## weighted by how strongly they vote for that digit. This is an
## approximation (it ignores ReLU) of what each digit "looks like" to the AI.
func _update_templates() -> void:
    var H := net.hidden
    var t := PackedFloat32Array(); t.resize(784 * 10)
    var w1 := net.w1
    var w2 := net.w2
    for i in 784:
        var base := i * H
        for h in H:
            var w := w1[base + h]
            var hb := h * 10
            for d in 10:
                t[d * 784 + i] += w * w2[hb + d]
    for d in 10:
        _template_textures[d].update(make_weight_image(t.slice(d * 784, (d + 1) * 784)))


# -------------------------------------------------------------- sort / select

func _sort_value(h: int) -> float:
    match sort_mode:
        SortMode.ACTIVATION:
            return acts[h] if acts.size() > h else 0.0
        SortMode.VOTE_FOR:
            return net.w2[h * 10 + sort_digit]
        SortMode.VOTE_AGAINST:
            return -net.w2[h * 10 + sort_digit]
        SortMode.STRENGTH:
            return strengths[h]
    return -h


func _apply_sort() -> void:
    if net == null:
        return
    var H := net.hidden
    var keys := PackedFloat32Array(); keys.resize(H)
    for h in H:
        keys[h] = _sort_value(h)
    var arr: Array = range(H)
    arr.sort_custom(func(a: int, b: int) -> bool:
        if keys[a] == keys[b]:
            return a < b
        return keys[a] > keys[b])
    order = PackedInt32Array(arr)
    for k in H:
        _flow.move_child(_thumbs[order[k]], k)
    _refresh_thumb_info()
    _count_label.text = "%d hidden neurons, sorted by: %s%s" % [
        H, SORT_NAMES[sort_mode],
        (" %d" % sort_digit) if sort_mode in [SortMode.VOTE_FOR, SortMode.VOTE_AGAINST] else ""]


func _refresh_thumb_info() -> void:
    var max_act := 0.0001
    for a in acts:
        max_act = maxf(max_act, a)
    for h in _thumbs.size():
        var a := acts[h] if acts.size() > h else 0.0
        var text := "#%d" % h
        match sort_mode:
            SortMode.VOTE_FOR, SortMode.VOTE_AGAINST:
                text += "  %+.2f" % net.w2[h * 10 + sort_digit]
            SortMode.STRENGTH:
                text += "  %.1f" % strengths[h]
            _:
                if a > 0.0:
                    text += "  %.2f" % a
        _thumbs[h].set_info(text, a / max_act)


func _select(h: int) -> void:
    if net == null or _thumbs.is_empty():
        return
    selected = clampi(h, 0, _thumbs.size() - 1)
    for t in _thumbs:
        t.set_pressed_no_signal(t.index == selected)
    _updating_ui = true
    _index_spin.value = selected
    _updating_ui = false
    _update_detail()
    _scroll.ensure_control_visible.call_deferred(_thumbs[selected])


func _step(dir: int) -> void:
    if order.is_empty():
        return
    var pos := order.find(selected)
    _select(order[posmod(pos + dir, order.size())])


func _unhandled_key_input(event: InputEvent) -> void:
    if not is_visible_in_tree() or not event.is_pressed():
        return
    var k := event as InputEventKey
    if k.keycode == KEY_RIGHT or k.keycode == KEY_DOWN:
        _step(1)
        accept_event()
    elif k.keycode == KEY_LEFT or k.keycode == KEY_UP:
        _step(-1)
        accept_event()


# ----------------------------------------------------------------- details

func _update_detail() -> void:
    if net == null:
        return
    var h := selected
    var H := net.hidden
    var w := net.neuron_weights(h)
    _big_tex.update(make_weight_image(w))
    _title.text = "Neuron #%d of %d" % [h, H]

    # input contribution map: pixel value x weight
    var has_input := pixels.size() == 784
    var raw := 0.0
    var contrib := PackedFloat32Array(); contrib.resize(784)
    if has_input:
        for i in 784:
            contrib[i] = pixels[i] * w[i]
            raw += contrib[i]
    _contrib_tex.update(make_weight_image(contrib))

    var votes := PackedFloat32Array(); votes.resize(10)
    var best := 0
    var worst := 0
    for d in 10:
        votes[d] = net.w2[h * 10 + d]
        if votes[d] > votes[best]: best = d
        if votes[d] < votes[worst]: worst = d
    var hl := sort_digit if sort_mode in [SortMode.VOTE_FOR, SortMode.VOTE_AGAINST] else -1
    _vote_bars.set_values(votes, hl)

    var a := acts[h] if acts.size() > h else 0.0
    var rank := 1
    for other in acts:
        if other > a:
            rank += 1
    var s := "Bias: %+.3f     Weight strength: %.2f\n" % [net.b1[h], strengths[h]]
    s += "Mostly votes FOR %d (%+.2f) and AGAINST %d (%+.2f)\n" % [best, votes[best], worst, votes[worst]]
    if has_input:
        s += "Your drawing: input sum %+.2f + bias = %+.2f\n" % [raw, raw + net.b1[h]]
        if a > 0.0:
            s += "Activation (ReLU): %.3f  -  rank %d of %d\n" % [a, rank, H]
            s += "-> adds %+.2f to digit %d's score right now" % [a * votes[best], best]
        else:
            s += "Activation (ReLU): 0  -  this neuron is silent for your drawing"
    else:
        s += "Draw a digit on the first tab to see how this neuron reacts."
    _stats.text = s


# ---------------------------------------------------------------------- UI

func _on_sort_changed(idx: int) -> void:
    sort_mode = idx as SortMode
    _sort_option.select(idx)
    _digit_spin.editable = sort_mode in [SortMode.VOTE_FOR, SortMode.VOTE_AGAINST]
    _apply_sort()
    _update_detail()


func _on_digit_changed(v: float) -> void:
    sort_digit = int(v)
    _apply_sort()
    _update_detail()


func _rank_by_digit(d: int) -> void:
    sort_digit = d
    sort_mode = SortMode.VOTE_FOR
    _sort_option.select(sort_mode)
    _digit_spin.set_value_no_signal(d)
    _digit_spin.editable = true
    _apply_sort()
    if not order.is_empty():
        _select(order[0])


func _build_ui() -> void:
    add_theme_constant_override("separation", 20)

    # ---------------- left: templates + controls + grid of neurons
    var left := VBoxContainer.new()
    left.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    left.add_theme_constant_override("separation", 8)
    add_child(left)

    var t_lbl := Label.new()
    t_lbl.text = "Digit templates - roughly what each digit looks like to the AI (click one to find the neurons that vote for it):"
    t_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    left.add_child(t_lbl)
    var t_row := HBoxContainer.new()
    t_row.add_theme_constant_override("separation", 4)
    left.add_child(t_row)
    var blank := Image.create(28, 28, false, Image.FORMAT_RGB8)
    for d in 10:
        var tex := ImageTexture.create_from_image(blank)
        _template_textures.append(tex)
        var b := Button.new()
        b.custom_minimum_size = Vector2(56, 72)
        b.focus_mode = Control.FOCUS_NONE
        b.tooltip_text = "Rank neurons by how strongly they vote for %d" % d
        var tr := TextureRect.new()
        tr.texture = tex
        tr.position = Vector2(4, 4)
        tr.size = Vector2(48, 48)
        tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
        tr.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
        tr.mouse_filter = Control.MOUSE_FILTER_IGNORE
        b.add_child(tr)
        var l := Label.new()
        l.text = str(d)
        l.position = Vector2(0, 52)
        l.size = Vector2(56, 18)
        l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
        l.mouse_filter = Control.MOUSE_FILTER_IGNORE
        b.add_child(l)
        b.pressed.connect(_rank_by_digit.bind(d))
        t_row.add_child(b)

    var c_row := HBoxContainer.new()
    c_row.add_theme_constant_override("separation", 8)
    left.add_child(c_row)
    var s_lbl := Label.new()
    s_lbl.text = "Sort by:"
    c_row.add_child(s_lbl)
    _sort_option = OptionButton.new()
    for n in SORT_NAMES:
        _sort_option.add_item(n)
    _sort_option.item_selected.connect(_on_sort_changed)
    c_row.add_child(_sort_option)
    var d_lbl := Label.new()
    d_lbl.text = "Digit:"
    c_row.add_child(d_lbl)
    _digit_spin = SpinBox.new()
    _digit_spin.min_value = 0
    _digit_spin.max_value = 9
    _digit_spin.editable = false
    _digit_spin.value_changed.connect(_on_digit_changed)
    c_row.add_child(_digit_spin)
    var hide_cb := CheckBox.new()
    hide_cb.text = "Hide unused border pixels"
    hide_cb.button_pressed = hide_unused
    hide_cb.focus_mode = Control.FOCUS_NONE
    hide_cb.tooltip_text = "Pixels that are (almost) never inked in MNIST never get trained,\nso their weights are just leftover random noise."
    hide_cb.toggled.connect(_on_hide_toggled)
    c_row.add_child(hide_cb)

    _count_label = Label.new()
    _count_label.modulate = Color(0.75, 0.75, 0.75)
    left.add_child(_count_label)

    _scroll = ScrollContainer.new()
    _scroll.size_flags_vertical = Control.SIZE_EXPAND_FILL
    _scroll.horizontal_scroll_mode = ScrollContainer.SCROLL_MODE_DISABLED
    left.add_child(_scroll)
    _flow = HFlowContainer.new()
    _flow.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    _flow.add_theme_constant_override("h_separation", 4)
    _flow.add_theme_constant_override("v_separation", 4)
    _scroll.add_child(_flow)

    # ---------------- right: details of the selected neuron
    var right := VBoxContainer.new()
    right.custom_minimum_size = Vector2(400, 0)
    right.add_theme_constant_override("separation", 8)
    add_child(right)

    _title = Label.new()
    _title.add_theme_font_size_override("font_size", 24)
    right.add_child(_title)

    var nav := HBoxContainer.new()
    right.add_child(nav)
    var prev := Button.new()
    prev.text = "< Prev"
    prev.pressed.connect(_step.bind(-1))
    nav.add_child(prev)
    var go_lbl := Label.new()
    go_lbl.text = "  Go to #"
    nav.add_child(go_lbl)
    _index_spin = SpinBox.new()
    _index_spin.min_value = 0
    _index_spin.max_value = 63
    _index_spin.value_changed.connect(func(v: float) -> void:
        if not _updating_ui:
            _select(int(v)))
    nav.add_child(_index_spin)
    var next := Button.new()
    next.text = "Next >"
    next.pressed.connect(_step.bind(1))
    nav.add_child(next)
    var keys_lbl := Label.new()
    keys_lbl.text = "  (arrow keys work too)"
    keys_lbl.modulate = Color(0.6, 0.6, 0.6)
    nav.add_child(keys_lbl)

    var imgs := HBoxContainer.new()
    imgs.add_theme_constant_override("separation", 12)
    right.add_child(imgs)
    _big_tex = ImageTexture.create_from_image(blank)
    _contrib_tex = ImageTexture.create_from_image(blank)
    _big_weights = _make_big_image(imgs, _big_tex, "Pattern (input weights)")
    _big_contrib = _make_big_image(imgs, _contrib_tex, "Your drawing x pattern")

    var legend := Label.new()
    legend.text = "Orange = wants ink there (positive weight)\nBlue = wants that pixel empty (negative weight)"
    legend.modulate = Color(0.8, 0.8, 0.8)
    right.add_child(legend)

    _stats = Label.new()
    _stats.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    right.add_child(_stats)

    var v_lbl := Label.new()
    v_lbl.text = "How this neuron votes for each digit (click a digit to rank all neurons by it):"
    v_lbl.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    right.add_child(v_lbl)
    _vote_bars = WeightBars.new()
    _vote_bars.custom_minimum_size = Vector2(0, 180)
    _vote_bars.bar_clicked.connect(_rank_by_digit)
    right.add_child(_vote_bars)


func _make_big_image(parent: Control, tex: Texture2D, caption: String) -> TextureRect:
    var box := VBoxContainer.new()
    parent.add_child(box)
    var tr := TextureRect.new()
    tr.texture = tex
    tr.custom_minimum_size = Vector2(190, 190)
    tr.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
    tr.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
    box.add_child(tr)
    var l := Label.new()
    l.text = caption
    l.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
    box.add_child(l)
    return tr
