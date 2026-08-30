# Draw this GPT architecture

One diagram. Top to bottom. Match this layout.

```
  token embedding          position embedding
       (gold)                  (gray)
         |                        |
         +------------⊕-----------+
                      |
        +-------------+-------------+
        | Block (x N)               |
        |             |             |
        |   layer normalization     |--
        |             |             |  |
        |   multi-head attention    |  | green
        |             |             |  |
        |            ⊕ <------------+  |
        |             |             |
        |   layer normalization     |--
        |             |             |  |
        |      feed forward         |  | green
        |             |             |  |
        |            ⊕ <------------+  |
        +-------------+-------------+
                      |
            layer normalization
                      |
                   Linear
                   (gold)
                      |
                   softmax
```

- Token embedding and position embedding side by side. Straight down, then bend into one circled `+`.
- Each module is its own box. Residual skips are green and land on circled `+`.
- `Block (x N)` on the block, not a second figure.
- Gold: token embedding and Linear. Gray: everything else.
- Legend: **Gold: token embedding and Linear share weights.**
