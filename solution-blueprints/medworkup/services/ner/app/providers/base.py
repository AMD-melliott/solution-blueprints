# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..schemas import Entity


class NERProvider(ABC):
    @abstractmethod
    def predict(self, text: str, labels: List[str], threshold: float) -> List[Entity]:
        """Run NER inference and return extracted entities sorted by start offset."""
        raise NotImplementedError
