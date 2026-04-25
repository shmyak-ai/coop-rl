---
description: "Use when writing or refactoring Python code in this project. Prefer a functional-first style (pure functions, composition, explicit data flow) while allowing object-oriented designs when they improve clarity, state management, or API boundaries."
name: "Python Functional-First Style"
applyTo: "**/*.py"
---
# Python Functional-First Style

- Prefer small, composable functions for core logic.
- Keep data flow explicit: pass inputs and return outputs rather than mutating shared state.
- Prefer pure functions where practical, especially in training logic, loss computation, and data transforms.
- Isolate side effects (I/O, logging, environment interaction, device placement, random seeding) at boundaries.
- Use plain data containers and typed structures when possible before introducing behavior-heavy classes.
- Object-oriented style is allowed and encouraged when it clearly improves encapsulation, lifecycle management, or readability.
- Do not force functional rewrites of well-structured class-based code; optimize for clarity and maintainability.
