extends SceneTree
# Full training test through the real UI flow (background thread):
# godot --headless --path . -s res://tests/train_test.gd

var main: Node
var started := false
var t0 := 0

func _initialize() -> void:
	main = load("res://scenes/main.tscn").instantiate()
	root.add_child(main)

func _process(_delta: float) -> bool:
	if not started:
		started = true
		t0 = Time.get_ticks_msec()
		main.epochs_spin.value = 1
		main._on_train_pressed()
		return false
	if main.train_btn.disabled:
		return false
	print("finished in ", (Time.get_ticks_msec() - t0) / 1000.0, " s")
	print("status: ", main.status_label.text)
	var c: DrawCanvas = main.canvas
	c._stamp_line(Vector2(140, 40), Vector2(140, 240))
	main._update_prediction()
	print("drawn '1' -> guess: ", main.guess_label.text, " ", main.confidence_label.text)
	return true
