# wpipln

`wpipln` (aka *weighted pipeline*) is a Python framework which helps you to model the data flow of your (weighted) data matrix `M` (`m x n`), label vector `y` (`m x 1`) and instance weight vector `w` (`m x 1`) through pipelines.

The API partially resembles [`sklearn.pipeline`](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) and thus paves the way for combining the frameworks.
The major difference between this framework and [`sklearn`](https://scikit-learn.org/stable/index.html) is, that each step, `transform` as well as `fit`, receives not only `X` and the label vector `y`, but also explicitly the instance weight vector `w`.

Below, we show a data flow diagram, where `(X, y, w)` is used for extracting parameters during `fit` that are subsequently used for transforming `(X', y', w')`. The methods `Fit #i` and `Tranform #i` are member functions of a the i-th pipeline step. Optionally, not all features (columns of `X`) are passed to the individual pipeline steps, but are filtered at selection branches `S╠` and subsequently merged with the remaining subset at merge branches `M╠`.
On top of that, pipelines can also filter instances (rows of `X`) during the fitting process. This filter is deactivated during the transforming step of the pipeline.
```
  ┌───────┐                         ┌──────────┐
  │X, y, w│                         │X', y', w'│          
  └───────┘                         └──────────┘
      ║                                  ║
+-----║--------------------+---+     +---║--------+---------+
|  ┌──────┐                |fit|     |   ║        |transform|
|  │Filter│                +---+     |   ║        +---------+
|  └──────┘                    |     |   ║                  |                 
|     ║                        |     |   ║                  |
|     ║              ┌──────┐  |     |   ║                  |
|    S╠════════╦═════│Fit #1│  |     |  S╠═════════╗        |
|     ║        ║     └───┬──┘  |     |   ║         ║        |
|     ║        ║         │     |     |   ║         ║        |
|     ║ ┌────────────┐   │     |     |   ║  ┌────────────┐  |
|     ║ │Transform #1├«──┴───────────────║─»┤Transform #1│  |
|     ║ └────────────┘         |     |   ║  └────────────┘  |
|     ║        ║               |     |   ║         ║        |
|     ║        ║               |     |   ║         ║        |
|    M╠════════╝               |     |  M╠═════════╝        |
|     ║              ┌──────┐  |     |   ║                  |
|    S╠════════╦═════│Fit #2│  |     |  S╠═════════╗        |
|     ║        ║     └───┬──┘  |     |   ║         ║        |
|     ║        ║         │     |     |   ║         ║        |
|     ║ ┌────────────┐   │     |     |   ║  ┌────────────┐  |
|     ║ │Transform #2├«──┴───────────────║─»┤Transform #2│  |
|     ║ └────────────┘         |     |   ║  └────────────┘  |
|     ║        ║               |     |   ║         ║        |
|     ║        ║               |     |   ║         ║        |
|    M╠════════╝               |     |  M╠═════════╝        |
|     :                        |     |   :                  |
|     :                        |     |   :                  |
+------------------------------+     +---║------------------+
                                         ║

Legend
******
Flow of (X, y, w):  ║, ═  
      Copy branch:  ╦
 Selection branch:  S╠
     Merge branch:  M╠
   Set parameters:  ──»─
```

## The Pipeline Step Interface

A pipeline step has to implement a `fit(X, y, w)` and a `transform(X, y, w)` member function. The former use `(X, y, w)` for setting properties (via side effect), whereas the latter is expected to be pure and transforms `(X, y, w)`. Further, pipelines offer the possibility to pass arguments to their individual steps. In order to work, pipeline steps are required to implement ` set_param(key, param)` and `set_params(self, params)`.
A good starting point for writing a custom pipeline step is to extend [BaseStep](wpipln/steps/BaseStep.py).

### Examples:
 - [`example.py`](example.py)
 - [`wpipln/test/helper.py`](wpipln/test/helper.py)


## The Pipeline Interface

A pipeline extends the interface of a pipeline step and thus can be used as a step in pipelines as well. Additionally, pipelines have a few more functionalities, such as the `filter(X, y, w)` member function which returns a selection that is later applied to `X`, `y` and `w` during the fitting procedure of the pipeline.
A good starting point for writing a custom pipeline step is to extend [Pipeline](wpipln/pipelines/Pipeline.py).

### Examples:
 - [`example.py`](example.py)
 - [`wpipln/pipelines/BalancedPipeline.py`](wpipln/pipelines/BalancedPipeline.py)