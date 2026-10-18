extends SceneTree
# Visual test of the pattern viewer (needs a window, not headless):
# godot --path . -s res://tests/viewer_test.gd -- <screenshot_dir>

var frames := 0
var main: Node
var out_dir := "user://"

func _initialize() -> void:
    var args := OS.get_cmdline_user_args()
    if args.size() > 0:
        out_dir = args[0]
    main = load("res://scenes/main.tscn").instantiate()
    root.add_child(main)

func _shot(name: String) -> void:
    var img := root.get_texture().get_image()
    img.save_png(out_dir.path_join(name))
    print("saved ", name)

func _process(_delta: float) -> bool:
    var c: DrawCanvas = main.canvas
    var pv: PatternViewer = main.pattern_viewer
    if pv.pixel_mask.is_empty() and frames < 3:
        return false  # wait for background mask computation
    frames += 1
    match frames:
        3:
            # draw a "7"
            c._stamp_line(Vector2(70, 60), Vector2(210, 60))
            c._stamp_line(Vector2(210, 60), Vector2(120, 240))
        6:
            _shot("tab1.png")
            var tabs: TabContainer = pv.get_parent()
            tabs.current_tab = 1
        10:
            _shot("tab2_index.png")
            pv._on_sort_changed(PatternViewer.SortMode.ACTIVATION)
            pv._select(pv.order[0])
        14:
            _shot("tab2_activation.png")
            pv._rank_by_digit(7)
        18:
            _shot("tab2_vote7.png")
            print("top neurons for 7: ", pv.order.slice(0, 5))
            print(pv._stats.text)
        20:
            return true
    return false
