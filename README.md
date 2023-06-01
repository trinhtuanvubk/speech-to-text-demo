## Speech to Text Demo with Wav2Vec2

### Setup

- To create Docker environment:
```
docker pull trinhtuanvubk/torch-w2v2:demofix
docker run -p 1430-1440:1430-1440 --add-host=host.docker.internal:host-gateway --restart always -itd -v $PWD/:/workspace --name torch-w2v2-demo -w/workspace trinhtuanvubk/torch-w2v2:demofix
docker exec -it torch-w2v2-demo bash
```

- To run demo app, run this command and open link on chrome: 
```
python3 app.py
```
