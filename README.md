# LUME-Torch

LUME-torch holds data structures used in the LUME modeling toolset. Variables and models built using LUME-torch will be compatible with other tools. LUME-torch uses [pydantic](https://pydantic-docs.helpmanual.io/) models to enforce typed attributes upon instantiation.

## Requirements

* Python >= 3.10
* pydantic
* numpy
* pyyaml
* mlflow

## Install

LUME-torch can be installed with conda using the command:

``` $ conda install lume-torch -c conda-forge ```

or through pip (coming soon):

``` $ pip install lume-torch ```

## Developer

A development environment may be created using the packaged `dev-environment.yml` file.

```
conda env create -f dev-environment.yml
```

Install as editable:

```
conda activate lume-torch-dev
pip install --no-dependencies -e .
```

Or by creating a fresh environment and installing the package:

```
pip install -e ".[dev]"
```

Note that this repository uses pre-commit hooks. To install these hooks, run:

```
pre-commit install
```

## Variables

The lume-torch variables are intended to enforce requirements for input and output variables by variable type. For now, only scalar variables (floats) are supported.

Minimal example of scalar input and output variables:

```python
from lume_torch.variables import ScalarVariable

input_variable = ScalarVariable(
    name="example_input",
    default_value=0.1,
    value_range=[0.0, 1.0],
)
output_variable = ScalarVariable(name="example_output")
```

All input variables may be made into constants by passing the
`is_constant=True` keyword argument. These constant variables are always
set to their default value and any other value assignments on
them will raise an error message.

## Models

The lume-torch base class `lume_torch.base.LUMEBaseModel` is intended to guide user development while allowing for flexibility and customizability. It is used to enforce LUME tool compatible classes for the execution of trained models.

Requirements for model classes:

* input_variables: A list defining the input variables for the model. Variable names must be unique. Required for use with lume-epics tools.
* output_variables: A list defining the output variables for the model. Variable names must be unique. Required for use with lume-epics tools.
* _evaluate: The evaluate method is called by the serving model.
  Subclasses must implement this method, accepting and returning a dictionary.

Example model implementation and instantiation:

```python
from lume_torch.base import LUMEBaseModel
from lume_torch.variables import ScalarVariable


class ExampleModel(LUMEBaseModel):
    def _evaluate(self, input_dict):
        output_dict = {
            "output1": input_dict[self.input_variables[0].name] ** 2,
            "output2": input_dict[self.input_variables[1].name] ** 2,
        }
        return output_dict


input_variables = [
    ScalarVariable(name="input1", default=0.1, value_range=[0.0, 1.0]),
    ScalarVariable(name="input2", default=0.2, value_range=[0.0, 1.0]),
]
output_variables = [
    ScalarVariable(name="output1"),
    ScalarVariable(name="output2"),
]

m = ExampleModel(input_variables=input_variables, output_variables=output_variables)
```

## Configuration files

Models and variables may be constructed using a YAML configuration file. The configuration file consists of three sections:

* model (optional, can alternatively pass a custom model class into the `model_from_yaml` method)
* input_variables
* output_variables

The model section is used for the initialization of model classes. The `model_class` entry is used to specify the model class to initialize. The `model_from_yaml` method will attempt to import the specified class. Additional model-specific requirements may be provided. These requirements will be checked before model construction. Model keyword arguments may be passed via the config file or with the function kwarg `model_kwargs`. All models are assumed to accept `input_variables` and `output_variables` as keyword arguments.

For example, `m.dump("example_model.yml")` writes the following to file

```yaml
model_class: ExampleModel
input_variables:
  input1:
    variable_class: ScalarVariable
    default_value: 0.1
    is_constant: false
    value_range: [0.0, 1.0]
  input2:
    variable_class: ScalarVariable
    default_value: 0.2
    is_constant: false
    value_range: [0.0, 1.0]
output_variables:
  output1: {variable_class: ScalarVariable}
  output2: {variable_class: ScalarVariable}
```

and can be loaded by simply passing the file to the model constructor:

```python
from lume_torch.base import LUMEBaseModel


class ExampleModel(LUMEBaseModel):
    def _evaluate(self, input_dict):
        output_dict = {
            "output1": input_dict[self.input_variables[0].name] ** 2,
            "output2": input_dict[self.input_variables[1].name] ** 2,
        }
        return output_dict


m = ExampleModel("example_model.yml")
```

## PyTorch Toolkit

A TorchModel can also be loaded from a YAML, specifying `TorchModel` in
the `model_class` of the configuration file.

```yaml
model_class: TorchModel
model: model.pt
output_format: tensor
device: cpu
fixed_model: true
```

In addition to the model_class, we also specify the path to the
TorchModel's model and transformers (saved using `torch.save()`).

The `output_format` specification indicates which form the outputs
of the model's `evaluate()` function should take, which may vary
depending on the application. TorchModel instances working with the
[LUME-EPICS](https://github.com/slaclab/lume-epics) service will
require an `OutputVariable` type, while [Xopt](https://github.
com/xopt-org/Xopt) requires either a dictionary of float
values or tensors as output.

The variables and any transformers can also be added to the YAML
configuration file:

```yaml
model_class: TorchModel
input_variables:
  input1:
    variable_class: ScalarVariable
    default_value: 0.1
    value_range: [0.0, 1.0]
    is_constant: false
  input2:
    variable_class: ScalarVariable
    default_value: 0.2
    value_range: [0.0, 1.0]
    is_constant: false
output_variables:
  output:
    variable_class: ScalarVariable
    value_range: [-.inf, .inf]
    is_constant: false
input_validation_config: null
output_validation_config: null
model: model.pt
input_transformers: [input_transformers_0.pt]
output_transformers: [output_transformers_0.pt]
output_format: tensor
device: cpu
fixed_model: true
precision: double
```

The TorchModel can then be loaded:

```python
from lume_torch.models.torch_model import TorchModel

# Load the model from a YAML file
torch_model = TorchModel("path/to/model_config.yml")
```


## TorchModule Usage

The `TorchModule` wrapper around the `TorchModel` is used to provide
a consistent API with PyTorch, making it easier to integrate with
other PyTorch-based tools and workflows.

### Initialization

To initialize a `TorchModule`, you need to provide the TorchModel object
or a YAML file containing the TorchModule model configuration.

```python
#  Wrap in TorchModule
torch_module = TorchModule(model=torch_model)

# Or load the model configuration from a YAML file
torch_module = TorchModule("path/to/module_config.yml")
```

### Model Configuration

The YAML configuration file should specify the `TorchModule` class
as well as the `TorchModel` configuration:

```yaml
model_class: TorchModule
input_order: [input1, input2]
output_order: [output]
model:
  model_class: TorchModel
  input_variables:
    input1:
      variable_class: ScalarVariable
      default_value: 0.1
      value_range: [0.0, 1.0]
      is_constant: false
    input2:
      variable_class: ScalarVariable
      default_value: 0.2
      value_range: [0.0, 1.0]
      is_constant: false
  output_variables:
    output:
      variable_class: ScalarVariable
  model: model.pt
  output_format: tensor
  device: cpu
  fixed_model: true
  precision: double
```

### Using the Model

Once the `TorchModule` is initialized, you can use it just like a
regular PyTorch model. You can pass tensor-type inputs to the model and
get tensor-type outputs.

```python
# Example input tensor
input_data = torch.tensor([[0.1, 0.2]])

# Evaluate the model
output = torch_module(input_data)

# Output will be a tensor
print(output)
```
### Saving using TorchScript

The `TorchModule` class' dump method has the option to save as a scripted JIT model by passing `save_jit=True` when calling the dump method. This will save the model as a TorchScript model, which can be loaded and used without the need for the original model file.

Note that saving as JIT through scripting has only been evaluated for NN models that don't depend on BoTorch modules.


## Logging

LUME-torch uses Python's standard `logging` module throughout the codebase to provide visibility into the library's operations. Logging configuration is left to the user application to allow maximum flexibility.

### Configuring Logging

To configure logging for your application using LUME-torch, set up the logging configuration in your application code:

```python
import logging

# Basic configuration - logs to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Now use lume-torch
from lume_torch.models.torch_model import TorchModel
torch_model = TorchModel("path/to/model_config.yml")
```

### Advanced Configuration

For more control over logging output, you can configure specific loggers:

```python
import logging

# Configure root logger
logging.basicConfig(level=logging.WARNING)

# Set specific log levels for lume-torch modules
logging.getLogger('lume_torch').setLevel(logging.INFO)
logging.getLogger('lume_torch.base').setLevel(logging.DEBUG)
logging.getLogger('lume_torch.models').setLevel(logging.INFO)
```

### Log Level Guidelines

LUME-torch follows standard Python logging practices with the following log levels:

* **DEBUG**: Detailed diagnostic information useful for troubleshooting
  - Module imports and initialization details
  - Variable parsing and serialization steps
  - Path resolution and file operations
  - Input/output transformation details

* **INFO**: General informational messages about normal operations
  - Model loading and initialization
  - File saving operations
  - MLflow model registration
  - Configuration file processing

* **WARNING**: Warning messages for potentially problematic situations
  - Deprecation warnings
  - Missing optional configurations
  - Fallback behaviors
  - JIT compilation limitations

* **ERROR**: Error messages logged before exceptions are raised
  - Validation failures
  - File not found errors
  - Invalid configurations
  - Type mismatches

### Example: Production Logging Setup

For production environments, you may want to log to a file with rotation:

```python
import logging
from logging.handlers import RotatingFileHandler

# Create formatters and handlers
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# File handler with rotation
file_handler = RotatingFileHandler(
    'lume_torch_app.log',
    maxBytes=10485760,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(formatter)

# Configure lume-torch logger
lume_logger = logging.getLogger('lume_torch')
lume_logger.setLevel(logging.INFO)
lume_logger.addHandler(file_handler)
lume_logger.addHandler(console_handler)
```

### Disabling Logging

If you want to suppress all LUME-torch logging:

```python
import logging

# Disable all lume-torch logging
logging.getLogger('lume_torch').setLevel(logging.CRITICAL)
```
