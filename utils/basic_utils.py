import os
import yaml

class Args:
    def __init__(self, hyp_dir):
        self.hyp_dir = hyp_dir  # hyp_dir을 인스턴스 변수로 저장
        self.hyps = self.load_config(hyp_dir)  # 설정 파일 로드
        for key, value in self.hyps.items():
            setattr(self, key, value)  # 객체에 key라는 변수를 생성하고, value를 값으로 할당함.
        
        # save_dir이 정의되어 있지 않으므로, 이를 self.hyps에서 추출하거나 기본값 설정
        self.save_dir = self.hyps.get("save_dir", "./saved_configs")
        
        self.make_dir(self.save_dir)
        self.save_config()  # save_config 메서드를 수정하여 인자 없이 호출

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    def save_config(self):
        # self.hyps와 self.save_dir을 사용하여 현재 설정 저장
        save_path = os.path.join(self.save_dir, "config.yaml")
        with open(save_path, 'w') as file:
            yaml.dump(self.hyps, file)
        print(f"Config saved to {save_path}")

    def make_dir(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs(f"{path}/imgs")
            os.makedirs(f"{path}/ckpt")
            os.makedirs(f"{path}/logs")

            print(f"{path} is generated.")
            
        else:
            print(f"{path} already exists.")