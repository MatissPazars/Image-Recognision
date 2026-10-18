class_name NeuralNetwork
extends RefCounted
## A small fully-connected neural network written in pure GDScript:
##   784 inputs (28x28 pixels) -> HIDDEN ReLU neurons -> 10 softmax outputs.
## Trained with stochastic gradient descent and cross-entropy loss.
##
## Speed trick: most MNIST pixels are 0, so the first layer only
## iterates over non-zero pixels (both forward and backward).

const INPUT := 784
const OUTPUT := 10

var hidden: int = 128
# w1 layout: [pixel * hidden + h]  (contiguous per pixel -> fast sparse access)
var w1 := PackedFloat32Array()
var b1 := PackedFloat32Array()
# w2 layout: [h * OUTPUT + o]
var w2 := PackedFloat32Array()
var b2 := PackedFloat32Array()

## Set to true from another thread to abort training early.
var stop_requested := false


func _init(hidden_size: int = 128) -> void:
    hidden = hidden_size
    randomize_weights()


func randomize_weights() -> void:
    w1.resize(INPUT * hidden)
    b1.resize(hidden)
    w2.resize(hidden * OUTPUT)
    b2.resize(OUTPUT)
    var std1 := sqrt(2.0 / INPUT)   # He initialisation for ReLU
    var std2 := sqrt(2.0 / hidden)
    for i in w1.size():
        w1[i] = randfn(0.0, std1)
    for i in w2.size():
        w2[i] = randfn(0.0, std2)
    b1.fill(0.0)
    b2.fill(0.0)


func duplicate_network() -> NeuralNetwork:
    var n := NeuralNetwork.new(1)
    n.hidden = hidden
    n.w1 = w1.duplicate()
    n.b1 = b1.duplicate()
    n.w2 = w2.duplicate()
    n.b2 = b2.duplicate()
    return n


# ---------------------------------------------------------------- inference

## pixels: 784 floats in range 0..1. Returns 10 probabilities.
func predict(pixels: PackedFloat32Array) -> PackedFloat32Array:
    var idx := PackedInt32Array()
    var val := PackedFloat32Array()
    for i in INPUT:
        if pixels[i] > 0.0:
            idx.append(i)
            val.append(pixels[i])
    return _forward(idx, val, w1, b1, w2, b2, PackedFloat32Array())


## Returns the ReLU activation of every hidden neuron for the given input.
func hidden_activations(pixels: PackedFloat32Array) -> PackedFloat32Array:
    var a1 := PackedFloat32Array(); a1.resize(hidden)
    if pixels.size() != INPUT:
        return a1
    var idx := PackedInt32Array()
    var val := PackedFloat32Array()
    for i in INPUT:
        if pixels[i] > 0.0:
            idx.append(i)
            val.append(pixels[i])
    _forward(idx, val, w1, b1, w2, b2, a1)
    return a1


## The 784 input weights of hidden neuron `h` (its "pattern"), in pixel order.
func neuron_weights(h: int) -> PackedFloat32Array:
    var out := PackedFloat32Array(); out.resize(INPUT)
    for i in INPUT:
        out[i] = w1[i * hidden + h]
    return out


## Predict directly from a MNIST image stored at `offset` inside a byte buffer.
func predict_bytes(images: PackedByteArray, offset: int) -> PackedFloat32Array:
    var idx := PackedInt32Array()
    var val := PackedFloat32Array()
    for i in INPUT:
        var p := images[offset + i]
        if p != 0:
            idx.append(i)
            val.append(p / 255.0)
    return _forward(idx, val, w1, b1, w2, b2, PackedFloat32Array())


func _forward(idx: PackedInt32Array, val: PackedFloat32Array,
        W1: PackedFloat32Array, B1: PackedFloat32Array,
        W2: PackedFloat32Array, B2: PackedFloat32Array,
        a1_out: PackedFloat32Array) -> PackedFloat32Array:
    var H := hidden
    var z1 := B1.duplicate()
    for k in idx.size():
        var x := val[k]
        var base := idx[k] * H
        for h in H:
            z1[h] += x * W1[base + h]
    for h in H:
        if z1[h] < 0.0:
            z1[h] = 0.0
    if a1_out.size() == H:
        for h in H:
            a1_out[h] = z1[h]
    var out := B2.duplicate()
    for h in H:
        var a := z1[h]
        if a == 0.0:
            continue
        var base := h * OUTPUT
        for o in OUTPUT:
            out[o] += a * W2[base + o]
    # softmax
    var m := out[0]
    for o in OUTPUT:
        m = maxf(m, out[o])
    var s := 0.0
    for o in OUTPUT:
        out[o] = exp(out[o] - m)
        s += out[o]
    for o in OUTPUT:
        out[o] /= s
    return out


static func argmax(v: PackedFloat32Array) -> int:
    var best := 0
    for i in v.size():
        if v[i] > v[best]:
            best = i
    return best


# ----------------------------------------------------------------- training

## Runs one epoch of SGD over the samples listed in `order`.
## progress_cb(done: int, total: int, running_accuracy: float) is called periodically.
## Returns the training accuracy over the epoch.
func train_epoch(images: PackedByteArray, labels: PackedByteArray,
        order: PackedInt32Array, lr: float, progress_cb: Callable) -> float:
    # Move arrays into locals and drop the member references so that writes
    # do not trigger copy-on-write duplication of the packed arrays.
    var W1 := w1; w1 = PackedFloat32Array()
    var B1 := b1; b1 = PackedFloat32Array()
    var W2 := w2; w2 = PackedFloat32Array()
    var B2 := b2; b2 = PackedFloat32Array()

    var H := hidden
    var a1 := PackedFloat32Array(); a1.resize(H)
    var dh := PackedFloat32Array(); dh.resize(H)
    var dz2 := PackedFloat32Array(); dz2.resize(OUTPUT)
    var idx := PackedInt32Array(); idx.resize(INPUT)
    var val := PackedFloat32Array(); val.resize(INPUT)
    var correct := 0
    var total := order.size()

    for n in total:
        if stop_requested:
            break
        var s := order[n]
        var offset := s * INPUT
        # gather non-zero pixels
        var nnz := 0
        for i in INPUT:
            var p := images[offset + i]
            if p != 0:
                idx[nnz] = i
                val[nnz] = p / 255.0
                nnz += 1
        var sidx := idx.slice(0, nnz)
        var sval := val.slice(0, nnz)

        var probs := _forward(sidx, sval, W1, B1, W2, B2, a1)
        var label := labels[s]
        if argmax(probs) == label:
            correct += 1

        # output gradient (softmax + cross-entropy)
        for o in OUTPUT:
            dz2[o] = probs[o]
        dz2[label] -= 1.0

        # hidden gradient and layer-2 update
        for h in H:
            var a := a1[h]
            var base := h * OUTPUT
            if a > 0.0:
                var g := 0.0
                for o in OUTPUT:
                    g += W2[base + o] * dz2[o]
                    W2[base + o] -= lr * a * dz2[o]
                dh[h] = g
            else:
                dh[h] = 0.0
        for o in OUTPUT:
            B2[o] -= lr * dz2[o]

        # layer-1 update (only rows of non-zero pixels)
        for h in H:
            dh[h] *= lr
            B1[h] -= dh[h]
        for k in nnz:
            var x := sval[k]
            var base := sidx[k] * H
            for h in H:
                W1[base + h] -= x * dh[h]

        if n % 1000 == 999 and progress_cb.is_valid():
            progress_cb.call(n + 1, total, float(correct) / (n + 1))

    w1 = W1; b1 = B1; w2 = W2; b2 = B2
    return float(correct) / maxi(1, total)


## Accuracy on a dataset (fraction 0..1).
func evaluate(images: PackedByteArray, labels: PackedByteArray, limit: int = -1) -> float:
    var count := labels.size() if limit < 0 else mini(limit, labels.size())
    var correct := 0
    for s in count:
        if stop_requested:
            return float(correct) / maxi(1, s)
        if argmax(predict_bytes(images, s * INPUT)) == labels[s]:
            correct += 1
    return float(correct) / maxi(1, count)


# ------------------------------------------------------------- persistence

func save(path: String, extra: Dictionary = {}) -> Error:
    var f := FileAccess.open(path, FileAccess.WRITE)
    if f == null:
        return FileAccess.get_open_error()
    var data := {
        "hidden": hidden, "w1": w1, "b1": b1, "w2": w2, "b2": b2, "extra": extra,
    }
    f.store_var(data)
    return OK


## Returns the "extra" dictionary on success, or null if loading failed.
func load_from(path: String) -> Variant:
    if not FileAccess.file_exists(path):
        return null
    var f := FileAccess.open(path, FileAccess.READ)
    if f == null:
        return null
    var data = f.get_var()
    if typeof(data) != TYPE_DICTIONARY or not data.has("w1"):
        return null
    hidden = data["hidden"]
    w1 = data["w1"]; b1 = data["b1"]; w2 = data["w2"]; b2 = data["b2"]
    return data.get("extra", {})
