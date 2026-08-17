# VIVA — Self-Examination

The project is not finished until every answer below is written out **from memory**, without
looking at code or notes, and then checked. These are the questions a senior will actually ask.

Mark each: `cannot answer` → `can answer with notes` → `can answer cold`.

---

## A. The whole pipeline

**A1. Walk one character from input to predicted next character, naming every tensor shape
along the way.**
Status: cannot answer
Answer:

**A2. Count the parameters of your model layer by layer, from the formulas.**
Status: cannot answer
Answer:

---

## B. Attention

**B1. What do Q, K and V mean, and why three separate projections instead of one?**
Status: cannot answer
Answer:

**B2. Why divide by √d_k specifically? What happens to the softmax when scores grow large?**
Status: cannot answer
Answer:

**B3. Why must the mask be applied before the softmax, and why −∞ rather than 0?**
Status: cannot answer
Answer:

**B4. Why do multiple heads help, when one head of the same total width does not?**
Status: cannot answer
Answer:

---

## C. The rest of the architecture

**C1. Where does the model see position, and what would it output without positional
information?**
Status: cannot answer
Answer:

**C2. Prove algebraically that two stacked linear layers with no nonlinearity between them
collapse into a single linear layer.**
Status: cannot answer
Answer:

**C3. What does a residual connection do to the gradient, algebraically?**
Status: cannot answer
Answer:

**C4. What does layer normalisation normalise *over*, and why that axis?**
Status: cannot answer
Answer:

---

## D. Mathematics and gradients

**D1. Derive the gradient of softmax + cross-entropy and show why it reduces to
`predicted − actual`.**
Status: cannot answer
Answer:

**D2. Explain the chain rule as a local rule on a computational graph.**
Status: cannot answer
Answer:

**D3. How does gradient checking work, and how do you choose `h`?**
Status: cannot answer
Answer:

**D4. Why does Adam beat SGD here? What do β₁, β₂ and ε control, and why is bias correction
necessary?**
Status: cannot answer
Answer:

---

## E. Engineering and honesty

**E1. Why is initial loss ≈ `ln(vocab_size)`, and what does it mean if it is not?**
Status: cannot answer
Answer:

**E2. Why is "overfit a single batch" the correct first training test?**
Status: cannot answer
Answer:

**E3. Why is this implementation thousands of times slower than PyTorch? Be specific.**
Status: cannot answer
Answer:

**E4. Your model's output is poor. Why, and what are the three cheapest ways to improve it?**
Status: cannot answer
Answer:
