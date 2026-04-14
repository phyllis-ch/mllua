# Nua
A small & simple single file library for implementing neural networks in Lua

> [!WARNING]
> Unstable software. Work in progress.

## Disclaimer
This is an amateur project.
The project was initiated to deepen my understanding in neural network and lua
Suggestions on code improvements are welcome.

## Requirements
- lua5.4 or higher

> lua5.3 and below is untested. Contributions are welcome.

## Usage
The main library provided functions to easily build up a neural network.

Before 12/4/2026, the library was seperated into matrix.lua and nua.lua.

There are many other programme can be coded using this library, 2 examples were given in this project.

matxor.lua demonstrated a xor gate using the functions provided by the library.

neuralxor.lua on the other hand, demonstrated a xor gate while also trained the model to determind whether the coordinate given is in a radius 1 circle centred at origin.

```lua
local nua = require("nua")

...
