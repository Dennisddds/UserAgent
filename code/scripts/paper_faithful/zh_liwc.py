# -*- coding: utf-8 -*-
"""Chinese LIWC-analog dictionary (shared lexical resource).

Both Semi-PerGCN (AAAI 2024) and TrigNet (ACL 2021) rely on the external
LIWC 2015 dictionary; this module is the Chinese stand-in for that resource.
Categories follow TrigNet's node scheme: 9 main psychological categories plus
6 personal-concern categories (15 total).
"""

import re

# 9 main psychological process categories
MAIN = {
    "posemo": "支持|赞|喜欢|高兴|感动|温暖|希望|振奋|自豪|骄傲|优秀|伟大|勇敢|真诚|可贵",
    "negemo": "愤怒|失望|担忧|痛心|可耻|恶劣|荒唐|愚蠢|悲剧|遗憾|讽刺|反感|警惕|焦虑",
    "cogproc": "认为|觉得|思考|理解|知道|相信|怀疑|判断|分析|原因|因为|所以|如果|应该",
    "social": "大家|我们|朋友|网友|人们|社会|公众|群众|一起|交流|讨论|评论",
    "percept": "看到|听说|发现|注意到|观察|显示|曝光|目睹|眼看",
    "bio": "健康|疾病|病毒|疫苗|身体|生命|医疗|医院|死亡|感染",
    "drives": "成功|成就|目标|竞争|赢|胜利|权力|地位|利益|荣誉|梦想",
    "relativ": "现在|过去|未来|时间|今天|明天|历史|年代|期间|阶段|地方|地区",
    "informal": "哈哈|呵呵|啊|吧|呀|嘛|哦|唉|哎|咱|老铁|吃瓜",
}
# 6 personal concern categories
CONCERN = {
    "work": "工作|就业|职场|加班|劳动|工资|裁员|岗位|职工|上班",
    "money": "钱|经济|收入|价格|投资|股市|财政|贸易|市场|消费|房价",
    "leisure": "旅游|电影|游戏|娱乐|假期|休闲|体育|球赛|音乐",
    "home": "家庭|父母|孩子|婚姻|家乡|房子|亲人|子女",
    "relig": "宗教|信仰|佛|上帝|教会|祈祷|神",
    "death": "死亡|去世|牺牲|遇难|逝世|悼念|丧生|殉职",
}

CATEGORIES = {**MAIN, **CONCERN}
_RX = {c: re.compile(p) for c, p in CATEGORIES.items()}


def word_categories(word):
    """LIWC lookup: categories a word belongs to."""
    return [c for c, rx in _RX.items() if rx.search(word)]


def text_categories(text):
    """Category match counts for a text."""
    return {c: len(rx.findall(text or "")) for c, rx in _RX.items()}
