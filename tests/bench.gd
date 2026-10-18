extends SceneTree
# Headless benchmark: godot --headless -s res://tests/bench.gd

func _init() -> void:
	var t0 := Time.get_ticks_msec()
	var imgs := MnistLoader.load_images(MnistLoader.TRAIN_IMAGES)
	var lbls := MnistLoader.load_labels(MnistLoader.TRAIN_LABELS)
	var timgs := MnistLoader.load_images(MnistLoader.TEST_IMAGES)
	var tlbls := MnistLoader.load_labels(MnistLoader.TEST_LABELS)
	print("loaded ", lbls.size(), " train / ", tlbls.size(), " test in ", Time.get_ticks_msec() - t0, " ms")
	var net := NeuralNetwork.new(64)
	var order := PackedInt32Array()
	for i in 6000:
		order.append(i)
	t0 = Time.get_ticks_msec()
	var acc := net.train_epoch(imgs, lbls, order, 0.02, Callable())
	print("train 6000: ", Time.get_ticks_msec() - t0, " ms, acc ", acc)
	t0 = Time.get_ticks_msec()
	print("test acc (2000): ", net.evaluate(timgs, tlbls, 2000), " in ", Time.get_ticks_msec() - t0, " ms")
	quit()
