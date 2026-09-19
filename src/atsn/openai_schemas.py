"""Strict JSON schemas for OpenAI Structured Outputs per pipeline stage."""

from __future__ import annotations

_CATEGORY = {"type": "string", "enum": ["CLOTHING", "FURNITURE"]}
_PRIMARY_SIGNAL = {"type": "string", "enum": ["TEXT", "IMAGE", "BOTH"]}
_ACCURACY_LABEL = {"type": "string", "enum": ["ACCURATE", "CORRECTED", "DELETED"]}
_ERROR_CATEGORY = {
    "anyOf": [
        {
            "type": "string",
            "enum": [
                "NONEXISTENT",
                "ATTRIBUTE_OR_OCR",
                "SPATIAL",
                "STATE",
                "ARTIFACT",
                "SUBJECTIVE",
            ],
        },
        {"type": "null"},
    ],
}
_COMPLETENESS_LABEL = {"type": "string", "enum": ["RETAINED", "ADDED"]}
_ADDITION_TYPE = {
    "anyOf": [
        {
            "type": "string",
            "enum": ["VISUAL_GAP", "NON_ENGLISH_TEXT", "KNOWN_ENTITY"],
        },
        {"type": "null"},
    ],
}
_REDUNDANCY_LABEL = {
    "type": "string",
    "enum": ["NOVEL", "UNAVOIDABLE_REPETITION", "AVOIDABLE_REDUNDANCY"],
}
_FUSION_SOURCE = {"type": "string", "enum": ["ORIGINAL", "ADDED"]}
_FUSION_ACTION = {
    "type": "string",
    "enum": [
        "RETAINED",
        "CORRECTED",
        "ADDED",
        "MERGED",
        "DROPPED_HALLUCINATION",
        "DROPPED_REDUNDANT",
    ],
}
_RELEVANCY_LABEL = {"type": "string", "enum": ["R", "S", "V"]}
_OBJECTIVITY_LABEL = {"type": "string", "enum": ["OBJECTIVE", "SUBJECTIVE"]}

SCHEMAS: dict[str, dict] = {
    "classification": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["classification"]},
            "category": _CATEGORY,
            "confidence": {"type": "number"},
            "primary_signal": _PRIMARY_SIGNAL,
            "text_image_agreement": {"type": "boolean"},
            "alternative_category": {
                "anyOf": [
                    {"type": "string", "enum": ["CLOTHING", "FURNITURE"]},
                    {"type": "null"},
                ],
            },
            "needs_review": {"type": "boolean"},
            "evidence": {"type": "string"},
        },
        "required": [
            "stage",
            "category",
            "confidence",
            "primary_signal",
            "text_image_agreement",
            "alternative_category",
            "needs_review",
            "evidence",
        ],
        "additionalProperties": False,
    },
    "claim_extraction": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["claim_extraction"]},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim": {"type": "string"},
                    },
                    "required": ["claim_id", "claim"],
                    "additionalProperties": False,
                },
            },
            "total_claims": {"type": "integer"},
        },
        "required": ["stage", "claims", "total_claims"],
        "additionalProperties": False,
    },
    "accuracy_validation": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["accuracy_validation"]},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "original_claim": {"type": "string"},
                        "claim": {"type": "string"},
                        "label": _ACCURACY_LABEL,
                        "error_category": _ERROR_CATEGORY,
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "original_claim",
                        "claim",
                        "label",
                        "error_category",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "total_claims": {"type": "integer"},
                    "accurate_claims": {"type": "integer"},
                    "corrected_claims": {"type": "integer"},
                    "deleted_claims": {"type": "integer"},
                    "error_rate": {"type": "number"},
                },
                "required": [
                    "total_claims",
                    "accurate_claims",
                    "corrected_claims",
                    "deleted_claims",
                    "error_rate",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["stage", "claims", "summary"],
        "additionalProperties": False,
    },
    "completeness_validation": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["completeness_validation"]},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "label": _COMPLETENESS_LABEL,
                        "addition_type": _ADDITION_TYPE,
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_id", "claim", "label", "addition_type", "reason"],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "input_claims": {"type": "integer"},
                    "retained_claims": {"type": "integer"},
                    "added_claims": {"type": "integer"},
                    "added_visual_gap": {"type": "integer"},
                    "added_non_english_text": {"type": "integer"},
                    "added_known_entity": {"type": "integer"},
                    "omission_score": {"type": "number"},
                },
                "required": [
                    "input_claims",
                    "retained_claims",
                    "added_claims",
                    "added_visual_gap",
                    "added_non_english_text",
                    "added_known_entity",
                    "omission_score",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["stage", "claims", "summary"],
        "additionalProperties": False,
    },
    "redundancy_validation": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["redundancy_validation"]},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "label": _REDUNDANCY_LABEL,
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_id", "claim", "label", "reason"],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "total_claims": {"type": "integer"},
                    "novel_claims": {"type": "integer"},
                    "unavoidable_repetition_claims": {"type": "integer"},
                    "avoidable_redundancy_claims": {"type": "integer"},
                    "redundancy_score": {"type": "number"},
                },
                "required": [
                    "total_claims",
                    "novel_claims",
                    "unavoidable_repetition_claims",
                    "avoidable_redundancy_claims",
                    "redundancy_score",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["stage", "claims", "summary"],
        "additionalProperties": False,
    },
    "fusion": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["fusion"]},
            "master_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "text": {"type": "string"},
                        "source": _FUSION_SOURCE,
                        "action": _FUSION_ACTION,
                    },
                    "required": ["claim_id", "text", "source", "action"],
                    "additionalProperties": False,
                },
            },
            "edit_log": {
                "type": "array",
                "items": {"type": "string"},
            },
            "final_alt_text": {"type": "string"},
        },
        "required": ["stage", "master_claims", "edit_log", "final_alt_text"],
        "additionalProperties": False,
    },
    "relevancy_evaluation": {
        "type": "object",
        "properties": {
            "stage": {"type": "string", "enum": ["relevancy_evaluation"]},
            "category": _CATEGORY,
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "label": _RELEVANCY_LABEL,
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_id", "claim", "label", "reason"],
                    "additionalProperties": False,
                },
            },
            "summary": {
                "type": "object",
                "properties": {
                    "total_claims": {"type": "integer"},
                    "relevant_claims": {"type": "integer"},
                    "staging_claims": {"type": "integer"},
                    "violation_claims": {"type": "integer"},
                    "w": {"type": "number"},
                    "lambda": {"type": "number"},
                    "relevancy_score": {
                        "anyOf": [{"type": "number"}, {"type": "null"}]
                    },
                    "note": {
                        "anyOf": [{"type": "string"}, {"type": "null"}]
                    },
                },
                "required": [
                    "total_claims",
                    "relevant_claims",
                    "staging_claims",
                    "violation_claims",
                    "w",
                    "lambda",
                    "relevancy_score",
                    "note",
                ],
                "additionalProperties": False,
            },
        },
        "required": ["stage", "category", "claims", "summary"],
        "additionalProperties": False,
    },
    "redundancy_evaluation": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "relevance_code": _RELEVANCY_LABEL,
                        "label": _REDUNDANCY_LABEL,
                        "dom_evidence": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                                "required": ["field", "value"],
                                "additionalProperties": False,
                            },
                        },
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "claim_id",
                        "claim",
                        "relevance_code",
                        "label",
                        "dom_evidence",
                        "reason",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
    "objectivity_evaluation": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_id": {"type": "string"},
                        "claim": {"type": "string"},
                        "label": _OBJECTIVITY_LABEL,
                        "reason": {"type": "string"},
                    },
                    "required": ["claim_id", "claim", "label", "reason"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["claims"],
        "additionalProperties": False,
    },
}


def schema_for_stage(stage_name: str) -> dict | None:
    """Return OpenAI response_format dict, or None for plain-text generation."""
    schema = SCHEMAS.get(stage_name)
    if schema is None:
        return None
    return {
        "type": "json_schema",
        "json_schema": {
            "name": stage_name,
            "strict": True,
            "schema": schema,
        },
    }
