import gradio as gr
from typing import Callable, Dict, List, Literal, TypedDict, Union


class Slider(TypedDict):
    type: Literal["slider"]
    label: str
    info: str
    min: float
    max: float
    default: float
    step: float


class Dropdown(TypedDict):
    type: Literal["dropdown"]
    label: str
    info: str
    choices: List[str]
    default: str
    value_type: Literal["value", "index"]


class Checkbox(TypedDict):
    type: Literal["checkbox"]
    label: str
    info: str
    default: bool


class Audio(TypedDict):
    type: Literal["audio"]
    label: str


class DeviceChooser(TypedDict):
    type: Literal["device_chooser"]


FormComponent = Union[Slider, Dropdown, Checkbox, Audio, DeviceChooser]


class FormDictItem(TypedDict):
    form: Dict[str, FormComponent]
    callback: Callable


class FormDictItemWithDynamicForm(TypedDict):
    form: Callable[..., Dict[str, FormComponent]]
    callback: Callable


class ParamInfo(TypedDict):
    model_name: str
    key: str
    comp: gr.component


FormDict = Dict[str, Union[FormDictItem, FormDictItemWithDynamicForm]]

FormDictInModelClass = Dict[str, FormComponent]
