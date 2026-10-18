extends SceneTree
# Smoke test: godot --headless --path . -s res://tests/smoke.gd
# Instantiates the main scene, draws a "1" on the canvas and runs a prediction.

var frames := 0
var main: Node

func _initialize() -> void:
	main = load("res://scenes/main.tscn").instantiate()
	root.add_child(main)

func _process(_delta: float) -> bool:
	frames += 1
	if frames == 2:
		var c: DrawCanvas = main.canvas
		c._stamp_line(Vector2(140, 40), Vector2(140, 240))
		var px := c.get_mnist_pixels()
		var s := ""
		for y in 28:
			for x in 28:
				s += "#" if px[y * 28 + x] > 0.5 else ("+" if px[y * 28 + x] > 0.0 else ".")
			s += "\n"
		print(s)
		main._update_prediction()
		print("guess: ", main.guess_label.text, " ", main.confidence_label.text)
		print("status: ", main.status_label.text)
	return frames > 3
