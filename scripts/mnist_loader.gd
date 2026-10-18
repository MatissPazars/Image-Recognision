class_name MnistLoader
extends RefCounted
## Reads MNIST IDX files (big-endian binary format).
## Images are returned as one flat PackedByteArray (count * 784 bytes),
## labels as a PackedByteArray (count bytes).

const ARCHIVE_DIR := "res://archive/"
const TRAIN_IMAGES := "train-images.idx3-ubyte"
const TRAIN_LABELS := "train-labels.idx1-ubyte"
const TEST_IMAGES := "t10k-images.idx3-ubyte"
const TEST_LABELS := "t10k-labels.idx1-ubyte"

const IMAGE_SIZE := 28
const PIXELS := IMAGE_SIZE * IMAGE_SIZE


static func load_images(file_name: String) -> PackedByteArray:
    var f := FileAccess.open(ARCHIVE_DIR + file_name, FileAccess.READ)
    if f == null:
        push_error("Cannot open %s (error %d)" % [file_name, FileAccess.get_open_error()])
        return PackedByteArray()
    f.big_endian = true
    var magic := f.get_32()
    if magic != 2051:
        push_error("%s is not an IDX image file (magic %d)" % [file_name, magic])
        return PackedByteArray()
    var count := f.get_32()
    var rows := f.get_32()
    var cols := f.get_32()
    if rows != IMAGE_SIZE or cols != IMAGE_SIZE:
        push_error("Unexpected image size %dx%d" % [rows, cols])
        return PackedByteArray()
    return f.get_buffer(count * PIXELS)


static func load_labels(file_name: String) -> PackedByteArray:
    var f := FileAccess.open(ARCHIVE_DIR + file_name, FileAccess.READ)
    if f == null:
        push_error("Cannot open %s (error %d)" % [file_name, FileAccess.get_open_error()])
        return PackedByteArray()
    f.big_endian = true
    var magic := f.get_32()
    if magic != 2049:
        push_error("%s is not an IDX label file (magic %d)" % [file_name, magic])
        return PackedByteArray()
    var count := f.get_32()
    return f.get_buffer(count)
