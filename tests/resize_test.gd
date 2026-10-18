extends SceneTree
# Checks the viewer handles a different hidden-neuron count (does not touch the saved model):
# godot --headless --path . -s res://tests/resize_test.gd

var frames := 0
var main: Node

func _initialize() -> void:
    main = load("res://scenes/main.tscn").instantiate()
    root.add_child(main)

func _process(_delta: float) -> bool:
    frames += 1
    var pv: PatternViewer = main.pattern_viewer
    if frames == 2:
        pv.set_network(NeuralNetwork.new(16))
    elif frames == 4:
        print("thumbs after 16: ", pv._thumbs.size(), " flow children: ", pv._flow.get_child_count())
        pv.set_network(NeuralNetwork.new(100))
    elif frames == 6:
        print("thumbs after 100: ", pv._thumbs.size(), " flow children: ", pv._flow.get_child_count())
        pv._on_sort_changed(PatternViewer.SortMode.STRENGTH)
        pv._step(1)
        print("selected after step: ", pv.selected, " title: ", pv._title.text)
        return true
    return false
