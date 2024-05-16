"""This module processes the incoming response from the miner"""

from bittensor import logging
from torch import Tensor
from copy import deepcopy
import healthi.base.utils as utils
import numpy as np

def calculate_distance_score(target, probs, label_weight) -> float:
    """This function calculates the distance score for a response

    The distance score is a result of the absolute distance for the
    response from each of the engine compared to the target value.
    The lower the distance the better the response is.

    Arguments:
        target:
            A float depicting the target confidence (0.0 or 1.0)

        engine_response:
            A dict containing the individual response produces by an
            engine

    Returns:
        distance:
            A dict containing the scores associated with the engine
    """

    if not utils.is_list_or_array(probs):
        return 0.0
    else:
        scores = (1 - np.abs(np.array(probs)-np.array(target))) * np.array(label_weight) # overall scores
        return scores



def calculate_total_distance_score(distance_scores: list) -> float:
    """Calculates the final distance score given all responses

    Arguments:
        distance_scores:
            A list of the distance scores

    Returns:
        total_distance_score:
            A float containing the total distance score used for the
            score calculation
    """
    if isinstance(distance_scores, bool) or not isinstance(distance_scores, list):
        return 0.0

    if distance_scores == []:
        return 0.0

    if len(distance_scores) > 1:
        total_distance_score = 1 - sum(distance_scores) / len(distance_scores)
    else:
        total_distance_score = 1 - distance_scores[0]

    return total_distance_score




def calculate_subscore_distance(response, target, label_weight) -> float:
    predicted_probs = response["predicted_probs"]

    distance_score = calculate_distance_score(target, predicted_probs, label_weight)
    return np.sum(distance_score)
    

def calculate_subscore_speed(timeout, response_time, label_weight):
    """Calculates the speed subscore for the response"""

    if isinstance(response_time, bool) or not isinstance(response_time, (float, int)):
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)):
        return None

    # If response time is 0.0 or larger than timeout, the time is invalid
    if response_time > timeout or response_time <= 0.0 or timeout <= 0.0:
        return None

    speed_score = 1.0 - (response_time / timeout)


    return speed_score * np.sum(label_weight)


def validate_response(hotkey, response) -> bool:
    """This method validates the individual response to ensure it has
    been format correctly

    Arguments:
        response:
            Response received from the miner

    Returns:
        outcome
            An instance of bool depicting the outcome of the validation.
    """
    # Responses without output are not valid
    if not response or isinstance(response, bool):
        logging.trace(f"Received an response without an output: {response}")
        return False

    # Check for type
    if not isinstance(response, dict):
        logging.trace(f"Received an response with incorrect type: {response}")
        return False

    # Check for mandatory keys
    mandatory_keys = [
        "predicted_probs",
        "task",
        "EHR",
        "subnet_version",
        "signature",
        "nonce",
        "timestamp",
    ]
    if not all(key in response for key in mandatory_keys):
        logging.trace(
            f"One or more mandatory keys: {mandatory_keys} missing from response: {response}"
        )
        return False

    # Check that the values are not empty
    for key in mandatory_keys:
        if response[key] is None:
            logging.trace(
                f"One or more mandatory keys: {mandatory_keys} are empty in: {response}"
            )
            return False

    # Check signature
    data = f'{response["nonce"]}{response["timestamp"]}'
    if not utils.validate_signature(
        hotkey=hotkey, data=data, signature=response["signature"]
    ):
        logging.debug(
            f'Failed to validate signature for the response. Hotkey: {hotkey}, data: {data}, signature: {response["signature"]}'
        )
        return False
    else:
        logging.debug(
            f'Succesfully validated signature for the response. Hotkey: {hotkey}, data: {data}, signature: {response["signature"]}'
        )

    # Check the validity of the confidence score
    if not utils.is_list_or_array(response["predicted_probs"]):
        logging.trace(f"predicted_probs is not correct type: {response['predicted_probs']}")
        return False

    # The response has passed the validation
    logging.trace(f"Validation succeeded for response: {response}")
    return True


def assign_score_for_uid(
    scores: Tensor, uid: int, alpha: float, response_score: float, sample_weight: float, label_weight
):
    """Assigns a score to an UID

    Arguments:
        scores:
            Current Tensor of scores
        uid:
            UID of the neuron to set the score for
        alpha:
            Scaling factor used for the degradation
        sample_weight:
            Weight of the current sample.

    Returns:
        scores:
            An updated Tensor of the scores
    """

    # Ensure the alpha is correctly defined
    if (
        not isinstance(alpha, float)
        or isinstance(alpha, bool)
        or alpha >= 1.0
        or alpha < 0.1
    ):
        logging.error(f"Value for alpha is incorrect: {alpha}")
        raise AttributeError(
            f"Alpha must be less than 1.0 and greater than or equal to 0.1. Value: {alpha}"
        )

    if (
        not isinstance(sample_weight, (int, float))
        or isinstance(sample_weight, bool)
        or not (0.0 < sample_weight <= 1.0)
    ):
        logging.error(f"Value for sample_weight is incorrect: {sample_weight}")
        raise AttributeError(
            f"sample_weight must be greater than or equal to 0.0 and less than or equal to 1.0. Value: {sample_weight}"
        )

    # Ensure the response score is correctly defined
    maxmum_scores = np.sum(label_weight)
    if not utils.validate_numerical_value(
        value=response_score, value_type=float, min_value=0.0, max_value=maxmum_scores
    ):
        logging.error(f"Value for response_score is incorrect: {response_score}")
        raise AttributeError(
            f"response_score must be in range (0.0, 1.0). Value: {response_score}"
        )

    # Ensure UID is correctly defined
    if not utils.validate_uid(uid):
        logging.error(f"Value for UID is incorrect: {uid}")
        raise AttributeError(f"UID must be in range (0, 255). Value: {uid}")

    old_score = deepcopy(scores[uid])

    # If current score is already at 0.0 we do not need to do anything
    if response_score == 0.0 and scores[uid] == 0.0:
        return scores, old_score, 0.0

    logging.trace(f"Assigning score for UID: {uid}. Current score: {old_score}")
    unweighted_new_score = alpha * scores[uid] + (1 - alpha) * response_score
    logging.trace(
        f"Unweighted score for UID: {uid}. Unweighted score: {unweighted_new_score}"
    )
    diff = unweighted_new_score - scores[uid]
    scores[uid] = scores[uid] + (sample_weight * diff)
    logging.trace(f"Assigned weighted score for UID: {uid}. New score: {scores[uid]}")

    if old_score == scores[uid]:
        logging.error(
            f"Score for UID: {uid} did not change. Old score: {old_score}, new score: {scores[uid]}"
        )
        raise ValueError(
            f"Score for UID: {uid} did not change. Old score: {old_score}, new score: {scores[uid]}"
        )

    return scores, old_score, unweighted_new_score.item()


def get_engine_response_object(
    total_score: float = 0.0,
    final_distance_score: float = 0.0,
    final_speed_score: float = 0.0,
    raw_distance_score: float = 0.0,
    raw_speed_score: float = 0.0,
) -> dict:
    """This method returns the score object. Calling the method
    without arguments returns default response used for invalid
    responses."""

    res = {
        "scores": {
            "total": total_score,
            "distance": final_distance_score,
            "speed": final_speed_score,
        },
        "raw_scores": {"distance": raw_distance_score, "speed": raw_speed_score},
    }

    return res


def get_response_object(
    uid: str, hotkey: str, target: list, EHR: list,
) -> dict:
    """Returns the template for the response object"""

    response = {
        "UID": uid,
        "hotkey": hotkey,
        "target": target,
        "EHR": EHR,
        "response": {},
        "scored_response": get_engine_response_object(),
        "weight_scores": {"new": 0.0, "old": 0.0, "change": 0.0, "unweighted": 0.0}
    }

    return response
