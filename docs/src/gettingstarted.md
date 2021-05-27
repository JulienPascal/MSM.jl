# Getting Started

Let's follow a learning-by-doing approach. As a warm-up, let's first estimate
parameters in serial. In a second step, we use several workers on a cluster.

## Example in serial

```@example
using MSM
using Plots

x = collect(range(-3.14, 4.0, length = 10))
y = sin.(x)
p = plot(x, y)
```

## Example in parallel

```@example
using MSM
using Plots

x = collect(range(-3.14, 4.0, length = 10))
y = cos.(x)
p = plot(x, y)
```
