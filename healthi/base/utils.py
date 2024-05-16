"""
This module implements common classes that are used by one or more core
features and their engines.
"""
import gc
import multiprocessing
import bittensor as bt
import numpy as np

def is_list_or_array(value):
    """Validates that a given value is a specific list or np array

    Arguments:
        value
            Value to validate
    
    Returns:
        result
            A bool depicting the outcome of the validation of certain 
            value format
    
    """

    if isinstance(value, list) and all(isinstance(x, float) for x in value):
        return 1
    elif isinstance(value, np.ndarray) and value.dtype.kind in 'f':
        return 1
    return 0


def validate_numerical_value(value, value_type, min_value, max_value):
    """Validates that a given value is a specific type and between the
    given range

    Arguments:
        value
            Value to validate
        type
            Python type  
        min
            Minimum value
        max
            Maximum value
    
    Returns:
        result
            A bool depicting the outcome of the validation
    
    """
    
    if isinstance(value, bool) or not isinstance(value, value_type):
        return False
    
    if (value < min_value) or (value > max_value):
        return False
    
    return True

def normalize_list(input_list: list) -> list:
    """
    This function normalizes a list so that values are between [0,1] and
    they sum up to 1.

    Arguments:
        input_list:
            A list containing values to be normalized. 
        
    Returns:
        normalized_list:
            A list instance. If input_list is length 1, the output will be [1.0]. 
            If there are negative values in input_list, the output will be normalized 
            between abs(min(input_list)) and the maximum value.
    """
    if len(input_list) > 1:
        min_val = min(input_list)

        if min_val >= 0:
            # If all values are non-negative, simply divide by sum
            sum_vals = sum(input_list)
            normalized_list = [val / sum_vals for val in input_list]
        else:
            # If there are negative values, normalize between abs(min) and max
            abs_min = abs(min_val)
            adjusted_values = [(val + abs_min) for val in input_list]
            sum_adjusted = sum(adjusted_values)
            normalized_list = [val / sum_adjusted for val in adjusted_values]
    else:
        normalized_list = [1.0]

    return normalized_list


def cleanup(variables: list = None):
    """
    This is a generic cleanup function
    
    Arguments:
        variables:
            A list containing variables to be cleaned.

    Returns:
        None
    """
    if variables:
        for variable in variables:
            variable = None
            del variable

    gc.collect()


def _run_function(result_dict, func, args, kwargs):
    """
    Helper function for the timeout() function
    
    Arguments:
        result_dict:
            A dictionary to which the results of the executed func 
            (specified by the func input) is appended.
        func:
            A function whose results will be appended to the inputted 
            result_dict.
    
    Returns:
        None    
    """
    result = func(*args, **kwargs)
    result_dict["result"] = result


def timeout_decorator(timeout):
    """
    Uses multiprocessing to create an arbitrary timeout for a
    function call. This function is used for ensuring a stuck function
    call does not block the execution of the neuron script

    Inputs:
        timeout:
            The amount of seconds to allow the function call to run 
            before timing out the execution. 

    Returns:
        decorator:
            A function instance which itself contains a subprocess wrapper().

    Raises:
        TimeoutError:
            Function call has timed out.    
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            manager = multiprocessing.Manager()
            result = manager.dict()

            process = multiprocessing.Process(
                target=_run_function, args=(result, func, args, kwargs)
            )
            process.start()
            process.join(timeout=timeout)

            if process.is_alive():
                process.terminate()
                process.join()
                raise TimeoutError(
                    f"Function '{func.__name__}' execution timed out after {timeout} seconds."
                )
            return result["result"]

        return wrapper

    return decorator


def validate_miner_blacklist(miner_blacklist) -> bool:
    """
    Checks that each entry in the inputted miner_blacklist list 
    is itself a JSON array with the flags 'hotkey' and 'reason'.

    An example of a valid input:
    [
        {
         "hotkey": "5FZV8fBTpEo51pxxPd5AqdpwN3BzK8rxog6VYFiGd6H7pPKY", 
         "reason": "Exploitation"
        },
        {
         "hotkey": "5FMjfXzFuW6wLYVGTrvE5Zd66T1dvgv3qKKhWeTFWXoQm3jS", 
         "reason": "Exploitation"
        }
    ]

    Arguments:
        miner_blacklist:
            A list instance representing the local miner blacklist--a 
            JSON array where each entry is a dict containing the keys 
            'hotkey' and 'reason'.
    
    Returns:
        bool:
            True if miner_blacklist is valid. False if not.
    """
    if miner_blacklist:
        return bool(
            isinstance(miner_blacklist, list)
            and all(
                isinstance(item, dict)
                and all(key in item for key in ["hotkey", "reason"])
                for item in miner_blacklist
            )
        )
    return False


def validate_uid(uid):
    """
    This method makes sure that a uid is an int instance between 0 and
    255. It also makes sure that boolean inputs are filtered out as
    non-valid uid's.

    Arguments:
        uid:
            A unique user id that we are checking to make sure is valid.
            (integer between 0 and 255).

    Returns:
        True:
            uid is valid--it is an integer between 0 and 255, True and
            False excluded.
        False:
            uid is NOT valid.
    """
    # uid must be an integer instance between 0 and 255
    if not isinstance(uid, int) or isinstance(uid, bool):
        return False
    if uid < 0 or uid > 255:
        return False
    return True

def validate_response_data(engine_response: dict) -> bool:
    """Validates the engine response contains correct data
    
    Arguments:
        engine_response:
            A dict containing the individual response produces by an
            engine
    
    Returns:
        result:
            A bool depicting the validity of the response
    """
    
    if isinstance(engine_response, bool) or not isinstance(engine_response, dict):
        return False
    
    required_keys = ["name", "value", "data"]
    for _,key in enumerate(required_keys):
        if key not in engine_response.keys():
            return False
        if engine_response[key] is None or engine_response[key] == "" or engine_response[key] == [] or engine_response[key] == {} or isinstance(engine_response[key], bool):
            return False
        
        if key == "value":
            if not validate_numerical_value(value=engine_response[key], value_type=float, min_value=0.0, max_value=1.0):
                return False
        
    return True

def validate_signature(hotkey: str, data: str, signature: str) -> bool:
    """Validates that the given hotkey has been used to generate the
    signature for data
    
    Arguments:
        hotkey:
            SS58_address of the hotkey used to sign the data
        data:
            Data signed
        signature:
            Signature of the signed data
    
    Returns:
        verdict:
            A bool depicting the validity of the signature
    """
    try:
        outcome = bt.Keypair(ss58_address=hotkey).verify(data, bytes.fromhex(signature))
        return outcome
    except AttributeError as e:
        bt.logging.error(f'Failed to validate signature: {signature} for data: {data} with error: {e}')
        return False
    except TypeError as e:
        bt.logging.error(f'Failed to validate signature: {signature} for data: {data} with error: {e}')
        return False
    except ValueError as e:
        bt.logging.error(f'Failed to validate signature: {signature} for data: {data} with error: {e}')
        return False

def sign_data(wallet: bt.wallet, data: str) -> str:
    """Signs the given data with the wallet hotkey
    
    Arguments:
        wallet:
            The wallet used to sign the Data
        data:
            Data to be signed
    
    Returns:
        signature:
            Signature of the key signing for the data
    """
    try:
        signature = wallet.hotkey.sign(data.encode()).hex()
        return signature
    except TypeError as e:
        bt.logging.error(f'Unable to sign data: {data} with wallet hotkey: {wallet.hotkey} due to error: {e}')
        raise TypeError from e
    except AttributeError as e:
        bt.logging.error(f'Unable to sign data: {data} with wallet hotkey: {wallet.hotkey} due to error: {e}')
        raise AttributeError from e

def validate_data(data_dict):

    # define valid data types for each key to check later
    # task, weight, hotkey, created_at, EHR, admission_time, label
    key_types = {
    'task':str,
    'admission_time':list,
    'EHR':list,
    'label':list,
    'weight': float,
    'hotkey': str,
    'created_at': str,
    'label_weight': list,
    }
    # run checks
    if not isinstance(data_dict, dict):
        bt.logging.info("Validate data failed with error as data not dict.")
        return False
    if len([pd for pd in data_dict]) != 8:
        keys = [pd for pd in data_dict]
        bt.logging.info(f"Validate data failed, with data keys error as {keys}.")
        return False
    for pd in data_dict:
        if pd not in ['task', 'weight', 'hotkey', 'created_at', 'EHR', 'admission_time', 'label', 'label_weight']:
            bt.logging.info(f"Validate data failed, {pd} not in keys")
            return False 
        if not isinstance(data_dict[pd], key_types[pd]):
            bt.logging.info(f"Validate data failed, {pd} mismatch key types")
            return False 
    return True