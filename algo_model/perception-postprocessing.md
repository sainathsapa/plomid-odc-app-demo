# Perception and post-processing

Implementation: `odc/segmentation.py`, `odc/attributes.py`, `odc/service.py`.

The service applies per-class floors, the operator threshold, class-aware NMS,
duplicate suppression, and vehicle/cabin hallucination filters. Persons and
generic COCO objects are retained as observations. A person detection alone is
not adult/child proof. A chair-like detection is only a restraint/object
candidate when the seat and occupancy evidence support it.

Fast masks are classical bbox/ROI refinement. SAM is an optional asynchronous
precise-mask branch whose output is tagged with its source frame. Neither mask
branch classifies a person or proves a safety state.

