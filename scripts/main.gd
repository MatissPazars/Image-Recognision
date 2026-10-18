extends Control
## Main screen: draw a digit, the neural network guesses it.
## Training runs on a background thread so the UI stays responsive.

const MODEL_PATH := "user://mnist_model.dat"
const HIDDEN_NEURONS := 64
const START_LR := 0.01
const LR_DECAY := 0.75

var net: NeuralNetwork
var model_ready := false

var _thread: Thread
var _train_net: NeuralNetwork
var _predict_dirty := false
var _train_start_ms := 0

# UI
var canvas: DrawCanvas
var preview_rect: TextureRect
var preview_image: Image
var preview_tex: ImageTexture
var guess_label: Label
var confidence_label: Label
var prob_bars: Array[ProgressBar] = []
var prob_labels: Array[Label] = []
var epochs_spin: SpinBox
var train_btn: Button
var stop_btn: Button
var reset_btn: Button
var train_progress: ProgressBar
var status_label: Label
var hidden_spin: SpinBox
var pattern_viewer: PatternViewer


func _ready() -> void:
    _build_ui()
    net = NeuralNetwork.new(HIDDEN_NEURONS)
    var extra = net.load_from(MODEL_PATH)
    if extra != null:
        model_ready = true
        var acc: float = extra.get("test_accuracy", 0.0)
        _set_status("Loaded saved model (test accuracy %.2f%%). Draw a digit!" % (acc * 100.0))
    else:
        _set_status("No trained model yet. Press \"Train\" to train on the MNIST archive.")
    hidden_spin.value = net.hidden
    pattern_viewer.set_network(net)
    _update_prediction()


func _process(_delta: float) -> void:
    if _predict_dirty:
        _predict_dirty = false
        _update_prediction()


func _exit_tree() -> void:
    if _thread and _thread.is_started():
        _train_net.stop_requested = true
        _thread.wait_to_finish()


# ---------------------------------------------------------------- prediction

func _on_drawing_changed() -> void:
    _predict_dirty = true


func _update_prediction() -> void:
    var pixels := canvas.get_mnist_pixels()
    pattern_viewer.set_input(pixels)
    # preview of what the network sees
    var bytes := PackedByteArray(); bytes.resize(784)
    for i in pixels.size():
        bytes[i] = int(pixels[i] * 255.0)
    preview_image.set_data(28, 28, false, Image.FORMAT_L8, bytes)
    preview_tex.update(preview_image)

    if pixels.is_empty() or not model_ready:
        guess_label.text = "?"
        confidence_label.text = "Draw a digit" if model_ready else "Train the model first"
        for i in 10:
            prob_bars[i].value = 0.0
            prob_labels[i].modulate = Color.WHITE
        return

    var probs := net.predict(pixels)
    var best := NeuralNetwork.argmax(probs)
    guess_label.text = str(best)
    confidence_label.text = "Confidence: %.1f%%" % (probs[best] * 100.0)
    for i in 10:
        prob_bars[i].value = probs[i]
        prob_labels[i].modulate = Color(0.4, 1.0, 0.5) if i == best else Color.WHITE


# ------------------------------------------------------------------ training

func _on_train_pressed() -> void:
    if _thread and _thread.is_started():
        return
    var wanted := int(hidden_spin.value)
    if wanted != net.hidden:
        # different size requested -> start a brand new network of that size
        net = NeuralNetwork.new(wanted)
        model_ready = false
        pattern_viewer.set_network(net)
        _update_prediction()
    _train_net = net.duplicate_network()
    _train_net.stop_requested = false
    _thread = Thread.new()
    _train_start_ms = Time.get_ticks_msec()
    train_btn.disabled = true
    reset_btn.disabled = true
    epochs_spin.editable = false
    hidden_spin.editable = false
    stop_btn.disabled = false
    train_progress.value = 0
    _thread.start(_train_thread.bind(int(epochs_spin.value)))


func _on_stop_pressed() -> void:
    if _train_net:
        _train_net.stop_requested = true
        _set_status("Stopping...")


func _on_reset_pressed() -> void:
    net = NeuralNetwork.new(int(hidden_spin.value))
    model_ready = false
    DirAccess.remove_absolute(ProjectSettings.globalize_path(MODEL_PATH))
    _set_status("Model reset (%d hidden neurons). Press \"Train\" to train a fresh network." % net.hidden)
    pattern_viewer.set_network(net)
    _update_prediction()


func _train_thread(epochs: int) -> void:
    _set_status.call_deferred("Loading MNIST data from archive/ ...")
    var images := MnistLoader.load_images(MnistLoader.TRAIN_IMAGES)
    var labels := MnistLoader.load_labels(MnistLoader.TRAIN_LABELS)
    var test_images := MnistLoader.load_images(MnistLoader.TEST_IMAGES)
    var test_labels := MnistLoader.load_labels(MnistLoader.TEST_LABELS)
    if labels.is_empty() or test_labels.is_empty():
        _set_status.call_deferred("Failed to load MNIST files from res://archive/")
        _on_training_finished.call_deferred()
        return

    var n := labels.size()
    var order := PackedInt32Array(); order.resize(n)
    for i in n:
        order[i] = i

    var lr := START_LR
    for e in epochs:
        # shuffle (Fisher-Yates)
        for i in range(n - 1, 0, -1):
            var j := randi() % (i + 1)
            var t := order[i]; order[i] = order[j]; order[j] = t

        var cb := func(done: int, total: int, acc: float) -> void:
            _on_progress.call_deferred(e, epochs, done, total, acc)
        var train_acc := _train_net.train_epoch(images, labels, order, lr, cb)
        if _train_net.stop_requested:
            break
        _set_status.call_deferred("Epoch %d/%d: testing on %d unseen images..." % [e + 1, epochs, test_labels.size()])
        var test_acc := _train_net.evaluate(test_images, test_labels)
        if _train_net.stop_requested:
            break
        _on_epoch_done.call_deferred(e, epochs, _train_net.duplicate_network(), train_acc, test_acc)
        lr *= LR_DECAY
    _on_training_finished.call_deferred()


func _on_progress(epoch: int, epochs: int, done: int, total: int, acc: float) -> void:
    var overall := (epoch + float(done) / total) / epochs
    train_progress.value = overall * 100.0
    var elapsed := (Time.get_ticks_msec() - _train_start_ms) / 1000.0
    var eta := elapsed / maxf(overall, 0.0001) - elapsed
    _set_status("Epoch %d/%d - image %d/%d - training accuracy %.1f%% - about %ds left" % [
        epoch + 1, epochs, done, total, acc * 100.0, int(eta)])


func _on_epoch_done(epoch: int, epochs: int, trained: NeuralNetwork, train_acc: float, test_acc: float) -> void:
    net = trained
    model_ready = true
    net.save(MODEL_PATH, {"test_accuracy": test_acc, "train_accuracy": train_acc})
    train_progress.value = (epoch + 1) * 100.0 / epochs
    _set_status("Epoch %d/%d done - train %.2f%% / test %.2f%% accuracy (model saved)" % [
        epoch + 1, epochs, train_acc * 100.0, test_acc * 100.0])
    pattern_viewer.set_network(net)
    _predict_dirty = true


func _on_training_finished() -> void:
    if _thread and _thread.is_started():
        _thread.wait_to_finish()
    if _train_net and _train_net.stop_requested:
        _set_status("Training stopped. " + ("Last completed epoch is kept." if model_ready else ""))
    train_btn.disabled = false
    reset_btn.disabled = false
    epochs_spin.editable = true
    hidden_spin.editable = true
    stop_btn.disabled = true


func _set_status(text: String) -> void:
    status_label.text = text


# ------------------------------------------------------------------------ UI

func _build_ui() -> void:
    var bg := ColorRect.new()
    bg.color = Color(0.11, 0.12, 0.15)
    bg.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    add_child(bg)

    var margin := MarginContainer.new()
    margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
    for side in ["left", "right", "top", "bottom"]:
        margin.add_theme_constant_override("margin_" + side, 24)
    add_child(margin)

    var root := VBoxContainer.new()
    root.add_theme_constant_override("separation", 16)
    margin.add_child(root)

    var title := Label.new()
    title.text = "MNIST Digit Recognizer - neural network in GDScript"
    title.add_theme_font_size_override("font_size", 26)
    root.add_child(title)

    var tabs := TabContainer.new()
    tabs.size_flags_vertical = Control.SIZE_EXPAND_FILL
    root.add_child(tabs)

    var hbox := HBoxContainer.new()
    hbox.add_theme_constant_override("separation", 32)
    tabs.add_child(hbox)
    tabs.set_tab_title(0, "Draw & Guess")

    pattern_viewer = PatternViewer.new()
    tabs.add_child(pattern_viewer)
    tabs.set_tab_title(1, "Neuron Patterns")

    # --- left: drawing
    var left := VBoxContainer.new()
    left.add_theme_constant_override("separation", 8)
    hbox.add_child(left)
    var hint := Label.new()
    hint.text = "Draw a digit (left mouse = draw, right = erase)"
    left.add_child(hint)
    canvas = DrawCanvas.new()
    canvas.custom_minimum_size = Vector2(392, 392)
    canvas.drawing_changed.connect(_on_drawing_changed)
    left.add_child(canvas)
    var clear_btn := Button.new()
    clear_btn.text = "Clear"
    clear_btn.pressed.connect(func() -> void: canvas.clear())
    left.add_child(clear_btn)

    var prev_row := HBoxContainer.new()
    prev_row.add_theme_constant_override("separation", 12)
    left.add_child(prev_row)
    preview_image = Image.create(28, 28, false, Image.FORMAT_L8)
    preview_tex = ImageTexture.create_from_image(preview_image)
    preview_rect = TextureRect.new()
    preview_rect.texture = preview_tex
    preview_rect.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
    preview_rect.custom_minimum_size = Vector2(112, 112)
    preview_rect.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
    prev_row.add_child(preview_rect)
    var prev_lbl := Label.new()
    prev_lbl.text = "What the AI sees\n(28x28, centred\nlike MNIST)"
    prev_row.add_child(prev_lbl)

    # --- right: prediction + training
    var right := VBoxContainer.new()
    right.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    right.add_theme_constant_override("separation", 8)
    hbox.add_child(right)

    var g_row := HBoxContainer.new()
    right.add_child(g_row)
    var g_title := Label.new()
    g_title.text = "AI guess: "
    g_title.add_theme_font_size_override("font_size", 32)
    g_row.add_child(g_title)
    guess_label = Label.new()
    guess_label.text = "?"
    guess_label.add_theme_font_size_override("font_size", 72)
    guess_label.add_theme_color_override("font_color", Color(0.4, 1.0, 0.5))
    g_row.add_child(guess_label)
    confidence_label = Label.new()
    right.add_child(confidence_label)

    var grid := GridContainer.new()
    grid.columns = 2
    grid.add_theme_constant_override("h_separation", 12)
    right.add_child(grid)
    for i in 10:
        var l := Label.new()
        l.text = str(i)
        l.add_theme_font_size_override("font_size", 20)
        grid.add_child(l)
        prob_labels.append(l)
        var bar := ProgressBar.new()
        bar.min_value = 0.0
        bar.max_value = 1.0
        bar.step = 0.0001
        bar.show_percentage = true
        bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
        bar.custom_minimum_size = Vector2(0, 24)
        grid.add_child(bar)
        prob_bars.append(bar)

    right.add_child(HSeparator.new())
    var t_title := Label.new()
    t_title.text = "Training (60,000 MNIST images, tested on 10,000 unseen ones)"
    right.add_child(t_title)

    var t_row := HBoxContainer.new()
    t_row.add_theme_constant_override("separation", 8)
    right.add_child(t_row)
    var e_lbl := Label.new()
    e_lbl.text = "Epochs:"
    t_row.add_child(e_lbl)
    epochs_spin = SpinBox.new()
    epochs_spin.min_value = 1
    epochs_spin.max_value = 50
    epochs_spin.value = 3
    t_row.add_child(epochs_spin)
    train_btn = Button.new()
    train_btn.text = "Train"
    train_btn.pressed.connect(_on_train_pressed)
    t_row.add_child(train_btn)
    stop_btn = Button.new()
    stop_btn.text = "Stop"
    stop_btn.disabled = true
    stop_btn.pressed.connect(_on_stop_pressed)
    t_row.add_child(stop_btn)
    reset_btn = Button.new()
    reset_btn.text = "Reset model"
    reset_btn.pressed.connect(_on_reset_pressed)
    t_row.add_child(reset_btn)

    var h_row := HBoxContainer.new()
    h_row.add_theme_constant_override("separation", 8)
    right.add_child(h_row)
    var h_lbl := Label.new()
    h_lbl.text = "Hidden neurons (patterns):"
    h_row.add_child(h_lbl)
    hidden_spin = SpinBox.new()
    hidden_spin.min_value = 4
    hidden_spin.max_value = 512
    hidden_spin.value = HIDDEN_NEURONS
    hidden_spin.tooltip_text = "Number of hidden neurons. Changing it starts a new network on Train/Reset.\nMore neurons = slightly better accuracy but slower training."
    h_row.add_child(hidden_spin)
    var h_hint := Label.new()
    h_hint.text = "(a new value starts a fresh network)"
    h_hint.modulate = Color(0.6, 0.6, 0.6)
    h_row.add_child(h_hint)

    train_progress = ProgressBar.new()
    train_progress.custom_minimum_size = Vector2(0, 24)
    right.add_child(train_progress)
    status_label = Label.new()
    status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    right.add_child(status_label)
