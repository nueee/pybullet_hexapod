# SNU URP21 hexapod project

in this repo, hexy pybullet env will be make based on simplecar

https://github.com/GerardMaggiolino/Gym-Medium-Post: urdf 기반의 단순한 gym env 및 TRPO 학습 example

여기서 [agent.py](http://agent.py) 는 SB3의 일부 기능에 해당하고, main.py는 train_first.py(우리가 쓰던)에 해당한다
여기서 주의 깊게 볼 것은 gym env를 만드는 구조로, [https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e](https://gerardmaggiolino.medium.com/creating-openai-gym-environments-with-pybullet-part-2-a1441b9a4d8e)
도 같이 참고해보면 이해가 빠를 것이다.

기본적인 디렉토리 구조의 개략도 :

    project_dir/
        setup.py (이 env를 패키지화해주어서, train_first 등의 py에서 import 가능하게 해줌)
        env_name/
            __ init __ .py (gym env에 본 env를 register해주는 과정, mujoco때도 했다.)
            envs/
                __ init __ .py (패키지화된 본 env에서 아래 파일의 class를 import)
                env_name_env .py (진짜 env. 여기서 최소한 __ init __ 함수, reset 함수, step 함수, render 함수, seed 함수, close 함수는 정의되어야 한다. 추가적인 함수 추가도 가능. 지금까지 mujoco에서 다루던 hexy_v4.env에 해당한다. 다만, 보다 직접적으로 더 많은 것을 지정해줘야 함. 오히려 장점?)
            rsc/
                __ init __ .py (필요없음)
                rsc1 .py (urdf를 불러오고, 필요하다면 action 부여도 가능하게)
                rsc1 .urdf
                rsc2 .py
                rsc2 .urdf
                ...

다만 이 구조를 만든다고 무조건 적용되는 것은 아니고, project dir/ 에서 pip install -e . 로 패키징해주어야 되는 듯하다. 또 추가적인 후처리가 필요할 수 있으므로 실천할 때 참고.

더이상 기존 라이브러리 파일을 수정하지 않고도, 별도의 디렉토리 관리가 가능해질 것이다.
이를 minitaur의 env 처럼 훨씬 복잡하게 커스터마이징하는 것도 물론 가능하다.
