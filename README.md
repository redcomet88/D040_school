# D040 Vue+Django+BERT深度学习校园百事通智能问答系统（大学新生智慧迎新版）

> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从github来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
> 
B站账号： **麦麦大数据**
编号：D040
## 1 视频

[video(video-mCkEfPOv-1741172328748)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=114109429454012)(image-https://i-blog.csdnimg.cn/img_convert/5a1ba73104375efcd6eebea93d45824e.jpeg)(title-BERT智能问答系统（大学新生迎新智慧迎新版）)]
## 2 系统简介和架构介绍
系统简介：本系统是一个基于Vue+Django+Bert模型构建的校园百事通深度学习新生问答系统，旨在为新生提供智能化的问答服务与校园信息支持。系统的核心功能围绕智能问答、公告管理、用户反馈和用户管理展开。主要包括：首页，用于展示系统概览和重要公告；智能问答模块，基于BERT模型计算文本相似度，为用户提供准确的问答服务，并支持会话保存及聊天记录管理；公告管理模块，用于发布和管理各类校园公告信息；用户反馈模块，收集并管理用户的反馈与建议，帮助系统不断改进；用户管理模块，支持用户注册、登录、权限分配以及个人设置（包括头像修改、个人信息更新和密码重置）。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e5acaac79dc04ebbb827470e088dce1b.png)
该系统采用典型的B/S（浏览器/服务器）架构模式。用户通过浏览器访问Vue前端界面，前端由HTML、CSS、JavaScript以及Vue.js生态系统中的Vuex（用于状态管理）、Vue Router（用于路由导航）和Element UI（用于界面组件化构建）等技术构建。前端通过RESTful API请求与Django后端进行交互，Django后端负责业务逻辑处理，并利用ORM（对象关系映射）技术与MySQL数据库进行数据持久化存储。系统还集成了BERT模型，用于智能问答功能，通过深度学习技术计算用户问题与知识库中答案的相似度，从而提供高效的问答服务。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/34673865d81a4aca97bd3fb491c1fa14.png)
## 3 功能说明
1 用户功能模块
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/1c44fc60914344beba287f4ef93a02ac.png)
公告和帮助：系统的一些信息会利用公告告诉用户
新生问答：问答功能基于BERT模型计算文本相似度
。(会话保存、聊天记录保存功能)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/80e2667ac02a4ce986deb54380ef8c89.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f2c4d5fe3ee94a608fe1aecef9910f92.png)

更改聊天记录：用户可以选择将自己的聊天记录进行删除，也可以选择添加一个新的聊天记录。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/81618a718bf94d3690a9caaa0a54ff2f.png)

用户反馈和建议：鼓励用户通过反馈和建议来帮助我们改进服务。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5f75a3f609f4efbbcf7c829e6ab15f2.png)

2 管理员功能
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b08f676136974c05be1326359e75fe1f.png)

用户管理：增删改查
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/10aa20193f5b4994b828e2f06ea6db44.png)

问题管理：问题和答案进行管理
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/65ac5a7aa14f4c4f93f20abc9ed23ca1.png)

用户反馈管理：回复用户的问题
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d2ede0a234d846f4840904c00dfe2544.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/384edf93ad3142f8ae14d84da28509a5.png)

更改公告：增删改查公告
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9297699f374843d095a971046c94d4a4.png)

其他
登录和注册
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7dd85bfd07ef4ec9b24c7b0eee8d403b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/add231c705bf4647b2b7c7a727164c10.png)
## 4 核心代码
### 4.1 代码说明
训练定制化模型：可以使用校园相关的数据微调BERT模型，提高匹配准确率。
多轮对话：增加对话历史记录，实现上下文理解。
前端交互：开发Web或移动端界面，方便用户提问。
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/071b0f6a9f634deb8539dc257d24f298.png)

### 4.3 核心代码
```python
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CampusQA:
    def __init__(self, model_path='bert-base-chinese'):
        # 加载预训练的BERT模型和分词器
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        # 知识库加载
        self.knowledge_base = self.load_knowledge_base()
    
    def load_knowledge_base(self):
        # 从数据库或文件加载校园知识数据
        # 假设知识库格式为 { "问题": "答案" }
        knowledge = {
            "什么是校园卡？": "校园卡是学生日常消费和进出校园的必备卡。",
            "如何报修校园设备？": "可以在校园维修系统在线提交报修申请。",
            # ... 更多知识对
        }
        return knowledge
    
    def get_embeddings(self, text):
        # 计算文本的BERT嵌入
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].numpy()[0]
    
    def answer_question(self, question):
        # 计算问题的嵌入
        q_embedding = self.get_embeddings(question)
        # 计算知识库中所有问题的相似度
        similarities = []
        for problem, answer in self.knowledge_base.items():
            p_embedding = self.get_embeddings(problem)
            similarity = np.dot(q_embedding, p_embedding)
            similarities.append((problem, similarity, answer))
        # 返回相似度最高的答案
        return max(similarities, key=lambda x: x[1])[2]

# 示例使用
qa = CampusQA()
print(qa.answer_question("校园卡的作用是什么？"))

```
