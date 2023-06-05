# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Frontend schema definitions."""
import json


def get_exp_config_schema() -> dict:
    """Utility function to get the experiment config json schema."""
    schema = """{
        "$schema": "http://json-schema.org/schema#7",
        "type": "object",
        "properties": {
            "general": {
                "type": "object",
                "properties": {
                    "logical_op": {
                        "type": "string",
                        "enum":  ["X", "Z"],
                        "minItems": 1
                    },
                    "qec_property": {
                        "type": "array",
                        "items": {
                            "enum": [
                                "logical_error_rate",
                                "threshold"
                            ]
                        }
                    },
                    "seed": {
                        "type": "integer"
                    }
                },
                "required": ["logical_op", "qec_property", "seed"]
            },
            "device": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                        "stim",
                        "clifford"
                        ]
                    },
                    "shots": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000000
                    }
                }
            },
            "code": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "RotatedPlanarCode",
                            "PlanarCode",
                            "ToricCode",
                            "RepetitionCode",
                            "ShorCode",
                            "FiveQubitCode"
                        ]
                    },
                    "size": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "rounds": {
                        "type": "integer",
                        "minimum": 1
                    },
                    "circuit_provided": {
                        "type": "boolean"
                    }
                },
                "required": [
                    "name",
                    "rounds"
                ]
            },
            "circuit": {
                "type": "object",
                "properties": {
                    "circuit_provided": {
                        "type": "boolean"
                    },
                    "has_errors": {
                        "type": "boolean"
                    },
                    "circuit_path": {
                        "type": "string"
                    }
                }
            },
            "errors": {
                "type": "object",
                "properties": {
                    "qubit_errors": {
                        "type": "object",
                        "items": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string"
                                },
                                "X": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean"
                                        },
                                        "distribution": {
                                            "type": "string",
                                            "enum": [
                                                "constant",
                                                "user",
                                                "gaussian"
                                            ]
                                        },
                                        "params": {
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "distribution"
                                    ]
                                },
                                "Y": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean"
                                        },
                                        "distribution": {
                                            "type": "string",
                                            "enum": [
                                                "constant",
                                                "user",
                                                "gaussian"
                                            ]
                                        },
                                        "params": {
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "distribution"
                                    ]
                                },
                                "Z": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean"
                                        },
                                        "distribution": {
                                            "type": "string",
                                            "enum": [
                                                "constant",
                                                "user",
                                                "gaussian"
                                            ]
                                        },
                                        "params": {
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "distribution"
                                    ]
                                },
                                "erasure": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean"
                                        },
                                        "distribution": {
                                            "type": "string",
                                            "enum": [
                                                "constant",
                                                "user",
                                                "gaussian"
                                            ]
                                        },
                                        "params": {
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "distribution"
                                    ]
                                },
                                "measurement": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean"
                                        },
                                        "distribution": {
                                            "type": "string",
                                            "enum": [
                                                "constant",
                                                "user",
                                                "gaussian"
                                            ]
                                        },
                                        "params": {
                                            "type": "array"
                                        }
                                    },
                                    "required": [
                                        "distribution"
                                    ]
                                }
                            },
                            "required": [
                                "data_path"
                            ]
                        }
                    },
                    "gate_errors": {
                        "type": "object",
                        "items": {
                            "type": "object",
                            "properties": {
                                "data_path": {
                                    "type": "string"
                                },
                                "correlated_pauli": {
                                    "type": "object",
                                    "properties": {
                                        "enabled": {
                                            "type": "boolean"
                                        }
                                    },
                                    "required": [
                                        "enabled"
                                    ]
                                }
                            },
                            "required": [
                                "correlated_pauli",
                                "data_path"
                            ]
                        }
                    }
                }
            },
            "decoder": {
                "type": "object",
                "properties" : {
                    "name" : {
                        "type": "string",
                        "enum": [
                            "PyMatchingDecoder",
                            "UnionFindDecoder",
                            "UnionFindNoWeights",
                            "FusionBlossomDecoder"
                        ]
                    },
                    "weighted": {
                        "type": "boolean"
                    }
                }
            }
        },
        "required": [
            "code",
            "errors"
        ]
    }
    """
    return json.loads(schema)
