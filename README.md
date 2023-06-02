## Speech to Text Demo with Wav2Vec2

### Setup

- To create Docker environment:
```
docker pull trinhtuanvubk/torch-w2v2:demofix
docker run -p 1430-1440:1430-1440 --add-host=host.docker.internal:host-gateway --restart always -itd -v $PWD/:/workspace --name torch-w2v2-demo -w/workspace trinhtuanvubk/torch-w2v2:demofix
docker exec -it torch-w2v2-demo bash
```

### Run Demo
- To run demo app, run this command and open link on chrome: 
```bash
bash run.sh
```
or 
```
python3 app.py \
--model_path "./model_repository/w2v2_ckpt/best_model.tar" \
--lm_path "./model_repository/language_model/4gram_small.arpa" \
--device 0 \
--port 1435 \
--use_language_model
```
- Flag:
    - `model_path`: path to finetuned model
    - `--lm_path`: path to language model
    - `--device`: select gpu device (`0`, `1`, ...). Remove this arg if using `cpu`
    - `--port`: port to run demo

