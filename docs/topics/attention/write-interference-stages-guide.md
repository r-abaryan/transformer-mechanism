# What happens to an answer while memory fills?

A history contains 32 key:value pairs. Each key occurs once. We write the whole history into one shared memory, then ask for each key's value. A training stage is a saved set of weights. A write position is a location inside one history. They are different axes.

The [experiment](write-interference-stages-results.md) follows exactly the same histories at initialization, step 3000 and step 12000. Endpoint errors and successful controls keep their identities across stages.

## The update and its two effects

Our memory has shape 16 by 8. With a normalized key k, value v and scalar write strength beta:

$$S_t=(I-\beta k_tk_t^T)S_{t-1}+\beta k_tv_t^T=A_tS_{t-1}+\beta k_tv_t^T.$$

Each write changes older memory through A and adds new content. A different key can therefore affect the answer to our query. Different keys can also carry the same value, so other writes are not always harmful.

For query q, zero-start memory gives

$$r_q=q^TS_T=\sum_t a_{qt}v_t^T,\qquad a_{qt}=\beta q^TA_T\cdots A_{t+1}k_t.$$

For its own write at position p, q = k_p. Its immediate coefficient is beta times squared key norm, approximately beta. The final coefficient divided by this immediate coefficient measures signed sensitivity to that write. It is **not the percentage of information retained**.

## From a memory read to an answer

The classifier is z = Wr + b. Fix c to the strongest non-answer class at the final frame. For true class y:

$$z_y-z_c=\underbrace{a_{qp}(W_y-W_c)v_p}_{\text{target write}}+\underbrace{\sum_{t<p}a_{qt}(W_y-W_c)v_t}_{\text{earlier writes}}+\underbrace{\sum_{t>p}a_{qt}(W_y-W_c)v_t}_{\text{later writes}}+b_y-b_c.$$

This exact decomposition distinguishes a weak target signal from competing content. It describes the actual computation; it does not explain why training selected those weights. For the immediate-versus-final correctness comparison, we use the best competing class at each frame. That competitor can change.

## What we learned

Most endpoint errors briefly had the right answer after their own write. Their final target contribution remains positive, but earlier and later contributions outweigh it on average. Training improves target coefficient retention while lowering beta; keys, values and classifier also change, so this comparison cannot credit beta alone.

Correct controls also lose answers temporarily and recover. Six last-position errors per seed have no later writes. Therefore neither temporary damage nor later-write interference explains every failure. The original loss supervises final answers; intermediate errors are not proof of a training defect.

Source inspected: [Yang et al. (2024), sections 2.2 and 3.1/3.2](https://arxiv.org/html/2406.06484v2). Delta updates and unrolling are published mathematics, written here in the transposed convention. The classifier decomposition is a local derivation; all scores are local measurements. Our tied-key scalar-beta reader is not the paper's full model. [Frozen methods](../../protocols/write-interference-training-stages.md).
