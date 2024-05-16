import typing
import bittensor as bt
import pydantic


class HealthiProtocol(bt.Synapse):
    """
    This class implements the protocol definition for the the
    healthi subnet.

    The protocol is a simple request-response communication protocol in
    which the validator sends a request to the miner for processing
    activities.
    """

    # Parse variables
    EHR: typing.Optional[list] = None
    output: typing.Optional[dict] = None

    synapse_nonce: str = pydantic.Field(
        ...,
        description="Synapse nonce provides an unique identifier for the data send out by the validator",
        allow_mutation=False
    )

    synapse_timestamp: str = pydantic.Field(
        ...,
        description="Synapse timestamp provides an unique identifier for the data send out by the validator",
        allow_mutation=False
    )

    subnet_version: int = pydantic.Field(
        ...,
        description="Subnet version provides information about the subnet version the Synapse creator is running at",
        allow_mutation=False,
    )

    task: str = pydantic.Field(
        ...,
        title="task",
        description="The task field provides instructions on which task to execute on the miner",
        allow_mutation=False,
    )

    synapse_signature: str = pydantic.Field(
        ...,
        title="synapse_signature",
        description="The synapse_signature field provides the miner means to validate the origin of the Synapse",
        allow_mutation=False,
    )

    def deserialize(self) -> bt.Synapse:
        """Deserialize the instance of the protocol"""
        return self
