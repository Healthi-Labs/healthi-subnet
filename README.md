<h1 align="center">Healthi Subnet</h1>

<p align="center">
  <a href="https://healthi.tech/">Website</a>
  ·
  <a href="https://twitter.com/Healthi_ai">Twitter</a>
    ·
  <a href="https://huggingface.co/Healthi">HuggingFace</a>
    .
  <a href="#FAQ">FAQ</a>

  ·  
</p>

# Introduction
This repository hosts the source code for the Healthi subnet, which operates atop the Bittensor network. The primary goal of this subnet is to utilize AI models for predictive diagnostics based on electronic health records (EHRs).

In the rapidly advancing field of healthcare technology, AI integration is transforming preventive medicine, particularly through predictive diagnostics. The increasing availability of patient data, especially EHRs, presents a significant opportunity to leverage AI for predicting health outcomes. This subnet on the Bittensor network rewards miners based on their AI models' performance in clinical prediction tasks, such as disease forecasting using EHRs. Our network aims to utilize these high-performing AI models developed by miners to improve patient outcomes, enhance healthcare delivery, and promote personalized clinical risk management.

# Quickstart
This repository requires Python 3.10 or higher and Ubuntu 22.04/Debian 12.

Installation (omit the first line if Bittensor is already installed):
```
$ /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/opentensor/bittensor/master/scripts/install.sh)"
$ sudo apt update && sudo apt install jq npm python3.10-dev python3.10-venv git && sudo npm install pm2 -g && pm2 update
$ git clone https://github.com/Healthi-Labs/healthi-subnet.git
$ cd healthi-subnet
$ python3 -m venv .venv
```

If you are not familiar with Bittensor, you should first perform the following activities:
- [Generate a new coldkey](https://docs.bittensor.com/getting-started/wallets#step-1-generate-a-coldkey)
- [Generate a new hotkey under your new coldkey](https://docs.bittensor.com/getting-started/wallets#step-2-generate-a-hotkey)

# Subnet register
Mainnet
```
btcli subnet register --netuid 34 --wallet.name {cold_wallet_name} --wallet.hotkey {hot_wallet_name}
```
Testnet:
```
btcli subnet register --netuid 133 --wallet.name {cold_wallet_name} --wallet.hotkey {hot_wallet_name} --subtensor.network test
```
> [!NOTE]  
> Validators need to establish an internet connection with the miner. This requires ensuring that the port specified in --axon.port is reachable on the virtual machine via the internet. This involves either opening the port on the firewall or configuring port forwarding.

If you want to run on testnet, set netuid to 133, and add --subtensor.network test. 

Run miner:
```
$ cd healthi-subnet
$ source .venv/bin/activate
$ bash scripts/run.sh \
--name healthi_miner \
--install_only 0 \
--max_memory_restart 10G \
--branch main \
--netuid 34 \
--profile miner \
--wallet.name {cold_wallet_name} \
--wallet.hotkey {hot_wallet_name} \
--axon.port 12345 
```

Run validator on testnet (validator updates automatically):
```
$ cd healthi-subnet
$ source .venv/bin/activate
$ bash scripts/run.sh \
--name healthi_validator \
--install_only 0 \
--max_memory_restart 5G \
--netuid 34 \
--profile validator \
--wallet.name {cold_wallet_name} \
--wallet.hotkey {hot_wallet_name}
```

To verify whether your miner is effectively responding to queries from validators, run the following commands (If you've recently started the miner, allow a few minutes):
```
cat ~/.pm2/logs/healthi-miner-out.log | grep "SUCCESS"
```
<h1 id="FAQ">FAQ</h1>

<details>
  <summary>How does rewarding work?</summary>
  <br>
  <p>
    Miners are rewarded based on the accuracy of their predictions for future health conditions derived from analyses of electronic health record (EHR) sequences. The top 20% of miners receive significantly higher rewards than the rest.
  </p>
</details>

<details>
  <summary>What is the expected data input and output as a miner?</summary>
  <br>
  <p>
    As a miner, your input will consist of sequences of Electronic Health Records (EHR) encoded with International Statistical Classification of Diseases and Related Health Problems (ICD-10) codes. In the following example, the patient visited the hospital twice, receiving two diagnoses each time:
    <br><br>
    <strong>Example Input:</strong>
    <pre>
[['D693', 'I10'], ['Z966', 'A047']]
    </pre>
    The current disease prediction task involves estimating the likelihood of getting the following 14 diseases within one year. Outputs should be an array or list of probabilities in the order listed below:
    <ol>
      <li>Hypertension</li>
      <li>Diabetes</li>
      <li>Asthma</li>
      <li>Chronic Obstructive Pulmonary Disease</li>
      <li>Atrial Fibrillation</li>
      <li>Coronary Heart Disease</li>
      <li>Stroke</li>
      <li>Anxiety and Depression</li>
      <li>Dementia</li>
      <li>Myocardial Infarction</li>
      <li>Chronic Kidney Disease</li>
      <li>Thyroid Disorder</li>
      <li>Heart Failure</li>
      <li>Cancer</li>
    </ol>
    <strong>Example Output:</strong>
    <pre>
[0.0027342219837009907, 0.012263162061572075, 0.01795087940990925, 0.016055596992373466, 0.010267915204167366, 0.0002267731324536726, 0.02317667566239834, 0.39082783460617065, 0.017462262883782387, 0.033581722527742386, 0.014757075347006321, 0.03425902500748634, 0.015123098157346249, 0.028889883309602737]
    </pre>
  </p>
</details>

<details>
  <summary>Compute Requirements</summary>
  <br>
  <p>
  The computational requirements for participating as a miner or validator in our subnet are minimal. Our subnet does not necessitate GPU capabilities and runs effectively on a virtual private server (VPS) with 4 virtual CPUs and 16 GB RAM. Although miners are permitted to use GPU resources, achieving higher rewards within our subnet depends more on developing superior predictive models than on computational power.
  </p>
</details>

<details>
  <summary>Data source and how do we prevent data exploitation?</summary>
  <br>
  <p>
Our data originates from authentic inpatient records, which are anonymized using Generative Adversarial Networks (GANs) to preserve the original data distributions while ensuring patient confidentiality. To prevent data exploitation and enhance security, our API continuously generates unique, synthetic electronic health record sequences for validators, protecting against replay attacks.
  </p>
</details>

<details>
  <summary>How can I train my model?</summary>
  <br>
  <p>
    Our base <a href="https://huggingface.co/Healthi/disease_prediction_v1.0">model</a> is a small Transformer model equipped with a customized tokenizer for electronic health record (EHR) data. We recommend miners train their model based on our tokenizer. Training data is available at <a href="https://github.com/Healthi-Labs/healthi-subnet/blob/main/healthi/base/data/trainset.parquet">this link</a>. Miners are also encouraged to use their own sourced EHR data for training.
  </p>
</details>

<details>
  <summary>Validator minimum staking requirements</summary>
  <br>
  <p>
    Validators need to stake at least 10,000 Tao on the mainnet or 10 Tao on the testnet to query our data API. Otherwise, validators can only acquire data locally for testing purposes. Data obtained locally carry significantly less weight than data from the API.
  </p>
</details>
