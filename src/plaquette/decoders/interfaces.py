# Copyright 2023, QC Design GmbH and the plaquette contributors
# SPDX-License-Identifier: Apache-2.0
"""Interface definitions for decoders.

The following interfaces implement all methods described in
:class:`.decoderbase.DecoderInterface`.
"""


from plaquette.decoders import (
    decoderbase,
    fusionblossom,
    matching,
    unionfind_interfaces,
)


class PyMatchingDecoder(decoderbase.DecoderInterface):
    """Min-weight perfect matching decoder provided by PyMatching.

    See :class:`.decoderbase.DecoderInterface` for supported methods.

    This is an interface to :class:`.matching.PyMatchingDecoder`.
    """

    _decoder_cls = matching.PyMatchingDecoder


class UnionFindNoWeights(decoderbase.DecoderInterface):
    """plaquette union find decoder, without support for weights.

    This decoder completely disabled processing weights within the union find
    algorithm.

    See :class:`.decoderbase.DecoderInterface` for supported methods.

    This is an interface to :class:`.unionfind_interfaces.UnionFindNoWeights`.
    """

    _decoder_cls = unionfind_interfaces.UnionFindNoWeights


class UnionFindDecoder(decoderbase.DecoderInterface):
    """plaquette union find decoder, with support for weights.

    .. deprecated:: 0.0.1a2
        Use the CPP implementation at https://github.com/qc-design/plaquette-unionfind
    """

    _decoder_cls = unionfind_interfaces.UnionFindDecoder


class FusionBlossomDecoder(decoderbase.DecoderInterface):
    """Min-weight perfect matching fast decoder provided by Fusion-Blossom.

    See :class:`.decoderbase.DecoderInterface` for supported methods.

    This is an interface to :class:`.fusionblossom.FusionBlossomDecoder`.
    """

    _decoder_cls = fusionblossom.FusionBlossomDecoder
