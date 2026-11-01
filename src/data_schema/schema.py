"""Unified sample format shared by converters, the server and the evaluator.

Every converter emits ``GuardrailSample`` rows; ``write_csv``/``read_csv`` are the
only functions that touch the CSV layout, so the on-disk format is defined here
and nowhere else.

CSV columns (``CSV_COLUMNS``)::

    sample_id, source, split, input_type, prompt, response, label, categories,
    language, attack_type, meta

* ``input_type`` mirrors LiteLLM's ``GenericGuardrailAPIRequest.input_type``:
  ``request`` = judge the user prompt (pre_call), ``response`` = judge the model
  output (post_call). For ``response`` rows the ``prompt`` is context only.
* ``label`` is the *ground truth for the judged side*.
* ``categories`` is ``|``-joined ``Category`` values (empty for safe rows).
* ``refusal_label`` (response rows only, optional): did the response refuse?
  ``refusal`` / ``partial`` / ``compliance`` — ground truth for validating refusal judges.
* ``meta`` is a JSON object string holding anything source-specific
  (original label, sub-category, toxicity score, ...).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Label(str, Enum):
    SAFE = "safe"
    UNSAFE = "unsafe"


class Category(str, Enum):
    """Unified taxonomy. Roughly MLCommons/Llama-Guard hazards + enterprise-specific risks.

    Converters / parsers map their native labels onto these. Keep the list short:
    every extra category is another slice that needs enough samples to be meaningful.
    """

    VIOLENCE = "violence"                    # violent crimes, threats, terrorism
    CRIME = "crime"                          # non-violent crimes, fraud, hacking, drugs
    WEAPONS_CBRN = "weapons_cbrn"            # weapons, chemical/biological/nuclear
    SEXUAL = "sexual"                        # adult sexual content
    CHILD_SAFETY = "child_safety"            # sexual content involving minors / CSAM
    HATE_HARASSMENT = "hate_harassment"      # hate speech, harassment, discrimination
    SELF_HARM = "self_harm"                  # suicide / self-harm / eating disorders
    PRIVACY_PII = "privacy_pii"              # personal data, doxxing
    CONFIDENTIAL = "confidential"            # company secrets / IP / internal-only data (enterprise)
    PROMPT_INJECTION = "prompt_injection"    # instructions hidden in input that try to hijack the system
    JAILBREAK = "jailbreak"                  # role-play / DAN-style policy-bypass attempts
    MISINFORMATION = "misinformation"        # elections, defamation, specialised advice
    OTHER = "other"                          # unsafe but not mappable


InputType = Literal["request", "response"]
RefusalLabel = Literal["refusal", "partial", "compliance"]
LANGUAGES = ("en", "zh-TW", "zh-CN", "zh", "ja", "ko", "mixed", "other", "unknown")

CSV_COLUMNS = [
    "sample_id",
    "source",
    "split",
    "input_type",
    "prompt",
    "response",
    "label",
    "categories",
    "language",
    "attack_type",
    "refusal_label",
    "meta",
]
OPTIONAL_COLUMNS = {"refusal_label"}  # older CSVs without these still load


class GuardrailSample(BaseModel):
    model_config = ConfigDict(use_enum_values=False, extra="forbid")

    sample_id: str = ""
    source: str
    split: str = "test"
    input_type: InputType = "request"
    prompt: str = ""
    response: str | None = None
    label: Label
    categories: list[Category] = Field(default_factory=list)
    language: str = "unknown"
    attack_type: str | None = None   # e.g. direct / indirect / roleplay / obfuscation / multi_turn
    refusal_label: RefusalLabel | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    # -- validation ------------------------------------------------------- #
    @field_validator("categories", mode="before")
    @classmethod
    def _split_categories(cls, v: Any) -> Any:
        if v is None or (isinstance(v, float) and v != v):  # NaN from pandas
            return []
        if isinstance(v, str):
            return [c for c in (s.strip() for s in v.split("|")) if c]
        return v

    @field_validator("meta", mode="before")
    @classmethod
    def _parse_meta(cls, v: Any) -> Any:
        if v in (None, ""):
            return {}
        if isinstance(v, str):
            return json.loads(v)
        return v

    @field_validator("language")
    @classmethod
    def _check_language(cls, v: str) -> str:
        if v not in LANGUAGES:
            raise ValueError(f"language must be one of {LANGUAGES}, got {v!r}")
        return v

    @model_validator(mode="after")
    def _consistency(self) -> "GuardrailSample":
        if self.input_type == "response" and not self.response:
            raise ValueError("input_type=response requires a non-empty response")
        if self.input_type == "request" and not self.prompt:
            raise ValueError("input_type=request requires a non-empty prompt")
        if self.label == Label.SAFE and self.categories:
            raise ValueError("safe samples must not carry unsafe categories")
        if self.refusal_label is not None and not self.response:
            raise ValueError("refusal_label needs a response")
        if not self.sample_id:
            self.sample_id = self.content_hash()
        return self

    # -- helpers ---------------------------------------------------------- #
    @property
    def text(self) -> str:
        """The text that is actually being judged."""
        return self.response if self.input_type == "response" else self.prompt  # type: ignore[return-value]

    @property
    def is_unsafe(self) -> bool:
        return self.label == Label.UNSAFE

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for part in (self.source, self.input_type, self.prompt, self.response or ""):
            h.update(part.encode("utf-8"))
            h.update(b"\x00")
        return f"{self.source}:{h.hexdigest()[:16]}"

    def to_row(self) -> dict[str, str]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "split": self.split,
            "input_type": self.input_type,
            "prompt": self.prompt,
            "response": self.response or "",
            "label": self.label.value,
            "categories": "|".join(c.value for c in self.categories),
            "language": self.language,
            "attack_type": self.attack_type or "",
            "refusal_label": self.refusal_label or "",
            "meta": json.dumps(self.meta, ensure_ascii=False, sort_keys=True) if self.meta else "",
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "GuardrailSample":
        row = {k: (None if v == "" else v) for k, v in row.items() if k in CSV_COLUMNS}
        for k in ("prompt",):
            row[k] = row.get(k) or ""
        return cls(**{k: v for k, v in row.items() if v is not None or k == "response"})


# --------------------------------------------------------------------------- #
# CSV IO
# --------------------------------------------------------------------------- #
def write_csv(samples: Iterable[GuardrailSample], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for s in samples:
            w.writerow(s.to_row())
            n += 1
    return n


def iter_csv(path: str | Path) -> Iterator[GuardrailSample]:
    csv.field_size_limit(1 << 30)
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = set(CSV_COLUMNS) - OPTIONAL_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for i, row in enumerate(reader):
            try:
                yield GuardrailSample.from_row(row)
            except Exception as e:  # surface the row number — debugging 50k-row CSVs is otherwise miserable
                raise ValueError(f"{path}: row {i + 2}: {e}") from e


def read_csv(path: str | Path) -> list[GuardrailSample]:
    return list(iter_csv(path))


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# language heuristics (no extra deps; swap for opencc / fasttext-langid if needed)
# --------------------------------------------------------------------------- #
_CJK = re.compile("[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")  # Ext-A, URO, compatibility ideographs
# hiragana + katakana letters; NOT U+30A0 ゠, U+30FB ・ (zh-TW writes 史蒂夫・賈伯斯) or U+30FC ー
_KANA = re.compile("[\u3040-\u309f\u30a1-\u30fa\u30fd-\u30ff]")
_HANGUL = re.compile(r"[가-힯]")
_LATIN = re.compile(r"[A-Za-z]")
# Characters that occur in only one Chinese script, generated from OpenCC's STCharacters / TSCharacters /
# TWVariants and restricted to the common-use tiers (GB2312 level 1 / Big5 常用字) plus every TW variant form.
# A Simplified character is listed only if OpenCC does not also use it as a Traditional character, and vice
# versa. That excludes characters that are standard in BOTH scripts with another meaning (公里 批准 皇后 松山
# 北斗 云云 范 划船 台北 只 干 系 面 制度 周 于 余 价 据 ...), which used to tag zh-TW text as zh-CN.
# 确 种 着 are added by hand: OpenCC keeps them as rare Traditional variants, but Taiwan text writes 確 種 著.
_TRAD_ONLY = frozenset(
    "丟並亂亙亞佇佈佔併來侖侶侷係俠倀倆倉個們倖倫偉側偵偽傑傖傘備傢傭傯傳債傷傾僅僑僕僥僱價儀儂億儈儉儐儔儕儘償優儲儷儸儻儼兇兌兒兗內"
    "兩冊冑冪凍凜凱別刪則剎剛剝剴創剷劃劇劉劊劍劑勁動務勛勝勞勢勣勳勵勸勻匯匱區協卹卻厭厲參叢吒吳吶呂員唸問啞啟喚喪喫喬單喲嗆嗇嗎嗚嗶"
    "嘆嘍嘔嘖嘗嘩嘮嘯嘰噓噥噯噴噸噹嚀嚇嚐嚕嚥嚨嚮嚴嚶囀囁囂囈囉囌囑囪國圍園圓圖團執堅堊堝堯報場塊塋塒塗塚塢塵塹墊墜墮墳墾壇壎壓壘壙壞"
    "壟壢壩壯壺壽夠夢夾奐奧奩奪奮妝姍姦娛婁婦媧媯媼媽嫗嫵嫻嬈嬋嬌嬤嬪嬰嬸孃孫學孿宮寢實寧審寫寬寵寶將專尋對導尷屆屍屜屢層屨屬岡峴島峽"
    "崑崗崙崢嵐嶄嶇嶔嶸嶺嶼嶽巒巔巖帥師帳帶幀幃幗幟幣幫幹幾庫廁廂廄廈廚廝廟廠廢廣廬廳弒弔張強彆彈彌彎彙彥彫彿後徑從徠復徹恆恥悅悵悶悽"
    "惡惱惻愛愜愴愷愾慄態慍慘慚慟慣慫慮慶慼慾憂憊憐憑憚憤憫憮憲憶懇應懍懣懲懶懷懸懺懼懾戀戰戲戶拋挾捨捫捱捲掃掄掙掛採揀揚換揮損搖搗搶"
    "摑摟摯摺摻撈撐撓撥撫撲撳撻撾撿擁擄擇擊擋擔據擠擬擰擱擲擴擷擺擻擾攆攏攔攙攜攝攣攤攪攬敗敘敵數斂斃斕斬斷昇時晉晝暈暉暘暢暫曄曆曇曉"
    "曖曠曬書會朧朮東枴柵桿梔條梟棄棗棟棧棲楊楓楨業極榦榮構槍槓槨槳樁樂樅樑樓標樞樣樸樹樺橈橋機橢橫檔檜檢檣檮檯檳檸檻櫃櫓櫚櫛櫝櫥櫬櫻"
    "欄權欐欖欽歎歐歟歡歲歷歸歿殘殤殮殯殲殺殼毀毆氈氣氫氬氳氾汎汙決沒沖況洩洶浹涇涼淒淚淨淪淵淺渙減渦測渾湊湧湯溈準溝溫溼滄滅滌滬滯滲"
    "滷滾滿漁漢漣漬漲漸漿潑潔潛潤潯潰澀澆澗澠澤澦澱濁濃濕濘濛濟濤濫濰濱濺濾瀆瀉瀋瀏瀕瀘瀝瀟瀨瀰瀲瀾灑灘灣灤災為烏無煉煙煥煩煬熒熱熾燈"
    "燉燒燙燜營燦燬燭燴燻燼燾爍爐爛爭爺爾牆牘牽犖犛犢犧狀狹狽猙猶獄獅獎獨獰獲獵獷獸獺獻玀現琺琿瑣瑤瑩瑪璣璦環璽璿瓊瓏瓔瓚甌甕產甦畝畢"
    "畫異當疇疊痙痠瘋瘍瘓瘡瘧瘺療癆癒癘癟癡癢癥癩癬癮癱癲發皚皰皺盃盜盞盡監盤盧盪眾睏睜睞瞞瞼矇矓矚矯硃硯碩確碼磚磧磯礎礙礦礪礫礬祕祿"
    "禍禎禦禪禮禱禿秈稅稈稜稟種稱穀穌積穎穠穡穢穩穫窩窪窮窯窺竄竅竇竊競筆筍筧箇箋箏節範築篠篤篩簍簑簞簡簣簫簽簾籃籌籟籠籤籬籮籲粵糝糞"
    "糧糰糾紀紂約紅紆紇紉紋納紐純紕紗紙級紛紜紡紮細紱紲紳紹紼絀終絃組絆結絕絛絞絡絢給絨絰統絲絳絹綁綏綑經綜綞綠綢綬維綰綱網綴綵綸綺綻"
    "綽綾綿緇緊緒緘緙線緝緞締緣編緩緬緯緲練緹緻縈縉縊縐縑縛縝縞縣縫縮縯縱縲縴縵縷縹總績繃繅繆繈繒織繕繚繞繡繩繪繫繭繳繹繼繽續纏纓纔纖"
    "纜缽罈罌罰罵罷羅羈羋羨義羶習翹聖聞聯聰聲聳聶職聽聾肅脅脈脣脩脫脹腎腦腫腳腸膚膠膩膽膾膿臉臍臏臘臚臟臢臥臨臺與興舉舊艙艦艱艷芻茲荊"
    "莊莖莢莧華菴菸萇萊萬萵葉葦葷蒍蒐蒞蒼蓀蓆蓋蓮蔔蔣蔥蔭蕩蕪蕭薊薑薔薦薩薺藍藝藥藪藷藹藺蘆蘇蘊蘋蘚蘭蘿處虛虜號虧蛻蜆蝕蝦蝨蝸螞螢螻蟈"
    "蟬蟯蟲蟻蠅蠍蠔蠟蠣蠱蠶蠻術衛衝袞裊補裝裡製複褲褸褻襖襠襤襪襬襯襲見規覓視覦親覬覲覺覽觀觴觸訂訃計訊訌討訏訐訑訓訕訖託記訛訝訟訢訣"
    "訥訪設許訴訶診註証詁詆詐詔評詖詛詞詠詢詣試詩詫詬詭詮詰話該詳詼誅誇誌認誑誕誘誚語誠誡誣誤誥誦誨說誰課誶誹誼調諂諄談諉請諍諒論諛諜"
    "諦諧諫諭諮諱諳諶諷諸諺諼諾謀謁謂謄謊謎謐謗謙講謝謠謨謫謬謹譁譆證譎譏識譙譚譜譟譫譯議譴護譽讀變讒讓讖讚讜豈豎豐豔豬貓貝貞負財貢貧"
    "貨販貪貫責貯貲貳貴貶買貸費貼貽貿賀賁賂賃賄賅資賈賊賑賒賓賜賞賠賡賢賣賤賦質賬賭賴賺賻購賽贅贈贊贍贏贓贖贗贛趕趙趨跡踐踴蹕蹟蹣蹤蹺"
    "躂躉躊躋躍躑躓躡躪軀車軋軌軍軏軒軔軛軟軸軻軼軾較載輊輒輓輔輕輛輜輝輞輟輥輦輩輪輯輳輸輻輾輿轂轄轅轉轍轎轔轟轡辦辭辮辯農迴逕這連週"
    "進遊運過達違遙遜遞遠適遲遷選遺遼邁還邇邊邏邐郵鄉鄒鄧鄭鄰鄴酈醃醜醞醣醫醬醱釀釁釅釋釐釗釘釙針釣釦釧釩釵鈇鈉鈍鈐鈑鈔鈕鈞鈣鈴鈷鈸鈹"
    "鈽鈾鈿鉀鉅鉉鉋鉍鉑鉗鉚鉛鉤鉸鉻銀銅銑銓銖銘銜銨銬銳銷銻銼鋁鋅鋇鋒鋤鋪鋰鋸鋼錄錐錕錘錙錚錠錡錢錦錨錫錮錯錳錶鍊鍋鍍鍔鍚鍛鍥鍬鍰鍵鍾"
    "鎂鎊鎔鎖鎘鎚鎢鎬鎮鎰鎳鏃鏈鏍鏑鏗鏘鏜鏝鏟鏡鏢鏤鏨鏽鐃鐘鐫鐮鐲鐳鐵鐸鐺鑄鑑鑒鑠鑣鑪鑰鑲鑷鑼鑽鑾鑿長門閂閃閉開閎閏閑閒間閔閘閡閣閤閥"
    "閨閩閭閱閻闆闈闊闋闌闐闔闕闖關闡闢陘陝陞陣陰陳陸陽隊階隕際隨險隱隴隸隻雋雖雙雛雜雞離難雲電霑霧霽靂靄靈靜靦靨鞏鞦韁韃韆韉韋韌韓韜"
    "韻響頁頂頃項順須頊頌預頑頒頓頗領頜頡頤頫頭頰頷頸頹頻顆題額顎顏顓願顛類顥顧顫顯顰顱風颯颱颳颶颺颼飄飛飢飩飪飭飯飲飴飼飽飾餃餅餉養"
    "餌餒餓餘餚餛餞餡館餵餾餿饅饉饑饒饗饜饞馬馭馮馱馳馴駁駐駑駒駕駙駛駝駟駢駭駱駿騁騎騖騙騫騰騵騷騾驀驃驅驍驕驗驚驛驟驢驥驪骯髏髒體髖"
    "髮鬆鬍鬚鬢鬥鬧鬨鬱魎魘魚魯魷鮑鮪鮫鮭鮮鯀鯉鯊鯖鯛鯧鯨鯰鯽鰍鰓鰥鰭鰱鰻鰾鱉鱔鱖鱗鱟鱷鱸鳥鳩鳳鳴鳶鴃鴆鴉鴒鴕鴛鴣鴦鴨鴻鴿鵑鵝鵠鵡鵪鵬"
    "鵲鶉鶯鶴鷂鷓鷗鷥鷹鷺鸚鸛鸞鹵鹹鹼鹽麗麥麩麴麵麼黃黌點黨黴黷鼕鼴齊齋齒齜齟齡齣齦齧齪齬齲齷龍龐龔龜"
)
_SIMP_ONLY = frozenset(
    "与专业丛东丝丢两严丧个临为丽举么义乌乐乔习乡书买乱争亏亚产亩亲亿仅从仑仓仪们众优会伞伟传伤伦伪体侠侣侥侦侧侨侩俩俭债倾偿储儿兑兰"
    "关兴兹养兽内冈册写军农冯冲决况冻净凄凉减凑凛凤凭凯击凿刘则刚创删别刹刽剂剐剑剥剧劝办务动励劲劳势勋匀区医华协单卖卢卤卧卫却厅历厉"
    "压厌厕厢厦厨厩县叁参双发变叙叠号叹吓吕吗吨听启吴呐呕员呛呜咏咙响哑哗哟唤啮啸喷嘘嘱嚣团园囱围国图圆圣场坏块坚坛坝坞坟坠垄垒垦垫堑"
    "堕墙壮声壳壶处备复够头夹夺奋奖奥妆妇妈娄娇娱婴婶孙学孪宁宝实宠审宪宫宽宾寝对寻导寿将尔尘尝尧尽层屉届属屡屿岁岂岗岛岭岿峡峦峰崭巩"
    "币帅师帐帜带帧帮幂并庄庆床庐库应庙庞废开异弃张弥弯弹强归当录彦彻径忆忧怀态怂怜总恋恒恳恶恼悦悬悯惊惧惨惩惫惭惮惯愤慑懒戏战户扑执"
    "扩扫扬扰抚抛抠抡抢护报担拟拢拣拥拦拧拨择挚挛挝挞挟挠挡挣挤挥捞损捡换捣掳掷掸掺揽搀搁搂搅携摄摆摇摈摊撑撵擞攒敌敛数斋斩断无旧时旷"
    "昼显晋晒晓晕暂术机杀杂权条来杨构枢枣枪枫柠栅标栈栋栏树栖样档桥桨桩梦检椭楼槛横樱橱欢欧歼残殴毁毕毙毡气氢汇汉汤汹沟没沤沥沦沧沪泞"
    "泪泻泼泽洁洒洼浅浆浇浊测济浑浓涛涝涟涡涣涤润涧涨涩渊渍渐渔渗温湾湿溃溅滚滞满滤滥滦滨滩潍潜澜濒灭灯灵灶灾灿炉点炼炽烁烂烃烛烟烦烧"
    "烩烫烬热焕爱爷牵牺犊状犹狈狞独狭狮狰狱猎猪猫献獭玛环现珐琐琼瑶瓮电画畅畴疗疟疡疮疯痈痉痒痪痴瘪瘫癣皑皱盏盐监盖盗盘着睁瞒瞩矫矾矿"
    "码砖砚砾础硕确硷碍碱礼祷祸禄离秃秆种秘积称秽税稳穷窃窍窑窜窝窥竖竞笋笔笺笼筛筹签简箩篓篮篱类粤粪粮紧纠红纤约级纪纫纬纯纱纲纳纵纶"
    "纷纸纹纺纽线练组绅细织终绊绍绎经绑绒结绕绘给绚络绝绞统绢绣绥绦继绩绪续绰绳维绵绷绸综绽绿缀缄缅缆缉缎缓缔缕编缘缚缝缠缨缩缮缴网罗"
    "罚罢羡群翘耸耻聂聋职联聪肃肠肤肮肾肿胀胁胆胶脉脏脐脑脓脚脱脸腻腾舆舰舱艰艳艺节芜芦苇苍苏茎茧荆荚荡荣荤荧荫药莱莲获莹萝萤营萧萨葱"
    "蒋蓝蓟蔷蔼蕴虏虑虚虱虽虾蚀蚁蚂蚕蛊蛮蛰蜕蜗蝇蝉衅衔补衬袄袜袭装裤见观规觅视览觉触誉誊计订讣认讥讨让讫训议讯记讲讳讶许讹论讼讽设访"
    "诀证评诅识诈诉诊诌词译试诗诚诛话诞诡询诣该详诧诫诬语误诱诲说诵请诸诺读诽课谁调谅谆谈谊谋谍谎谐谓谗谚谜谢谣谤谦谨谩谬谭谰谱谴贝贞"
    "负贡财责贤败账货质贩贪贫贬购贮贯贰贱贴贵贷贸费贺贼贾贿赁赂赃资赊赋赌赎赏赐赔赖赘赚赛赞赠赡赢赣赵赶趋跃践踌踪蹿躯车轧轨轩转轮软轰"
    "轴轻载轿较辅辆辈辉辊辐辑输辕辖辗辙辞辩辫边辽达迁过迈运还这进远违连迟迹选逊递逻遗遥邓邮邹邻郑郧郸酝酱酿释鉴针钉钎钒钓钙钝钞钟钠钡"
    "钢钥钦钧钨钩钮钱钳钵钻钾铀铁铂铃铅铆铜铝铡铣铬铭铰铱铲银铸铺链销锁锄锅锈锋锌锐锑锗错锚锡锣锤锥锦锨锭键锯锰锹锻镀镁镇镊镍镐镑镜镣"
    "镭镰镶长门闪闭问闯闰闲间闷闸闹闺闻闽阀阁阂阅阉阎阐阑阔队阳阴阵阶际陆陇陈陕陨险随隐隶难雇雏雾霉静韦韧韩韵页顶顷项顺须顽顾顿颁颂预"
    "颅领颇颈颊颐频颓颖颗题颜额颠颤颧风飘飞饥饭饮饯饰饱饲饵饶饺饼饿馁馅馆馈馋馏馒马驭驮驯驰驱驳驴驶驹驻驼驾骂骄骆骇骋验骏骑骗骚骡骤鱼"
    "鲁鲍鲜鲤鲸鳃鳖鳞鸟鸡鸣鸥鸦鸭鸯鸳鸵鸽鸿鹃鹅鹊鹏鹤鹰麦黄齐齿龄龋龙龚龟"
)
ZH_LANGUAGES = frozenset({"zh", "zh-TW", "zh-CN"})  # the Chinese members of LANGUAGES


# Taiwan-standard forms that OpenCC's Simplified tables also produce (TWVariants targets / Big5 common):
# they say nothing about the script, so they must not count as Simplified markers.
_SIMP_ONLY = _SIMP_ONLY - frozenset("群床峰秘痴灶雇霉")
# Japanese shinjitai not used in either Chinese script (TW 氣/圖/驛 vs CN 气/图/驿)
_JA_ONLY = frozenset("気図駅広売読辺険鉄楽薬戦転対単処発総権歳様沢桜変恋")
# Traditional markers that are also everyday Japanese kanji: not evidence of Chinese next to a kana
_TRAD_SHARED_WITH_JA = frozenset("東")


def detect_language(text: str | None) -> str:
    """Cheap script-based guess: ``en`` / ``zh-TW`` / ``zh-CN`` / ``zh`` (script unknown) / ``ja`` / ``ko`` /
    ``mixed`` / ``unknown``. Latin text is always ``en``. Converters that KNOW the language (a ``lang`` column,
    a per-language file, OpenCC output) must pass it instead of relying on this."""
    if not text:
        return "unknown"
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    kana = len(_KANA.findall(text))
    # one stray kana (zh-TW loves 「小確幸の…」) is not Japanese unless nothing else marks the text as Chinese
    # (東京の天気 / 大阪の人口 are Japanese); real Japanese usually has particles everywhere
    if kana and kana > 0.1 * max(cjk, 1):
        if kana >= 2 or not cjk or any(ch in _JA_ONLY for ch in text):
            return "ja"
        if not any((ch in _TRAD_ONLY and ch not in _TRAD_SHARED_WITH_JA) or ch in _SIMP_ONLY for ch in text):
            return "ja"
    if len(_HANGUL.findall(text)) > 5:
        return "ko"
    if cjk == 0:
        return "en" if latin else "unknown"
    if latin > 3 * cjk:
        return "mixed" if cjk >= 5 else "en"
    trad = sum(ch in _TRAD_ONLY for ch in text)
    simp = sum(ch in _SIMP_ONLY for ch in text)
    if trad > simp:
        return "zh-TW"
    if simp > trad:
        return "zh-CN"
    return "zh"


# region / script spellings seen in exports -> our LANGUAGES
_LANGUAGE_ALIASES = {
    "zh-tw": "zh-TW", "zh-hant": "zh-TW", "zh-hant-tw": "zh-TW", "zh_hant": "zh-TW",
    "zh-cn": "zh-CN", "zh-hans": "zh-CN", "zh-hans-cn": "zh-CN", "zh-sg": "zh-CN", "zh_hans": "zh-CN",
    "zh": "zh", "zh-hk": "zh", "zh-mo": "zh", "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
}


def normalize_language(code: Any) -> str | None:
    """Map a dataset's own language code onto ``LANGUAGES``: ``None`` for a missing value (None / NaN / blank /
    ``auto``, i.e. "detect it"), the matching code (case- and ``_``-insensitive, ``en-US`` -> ``en``), else ``other``
    (``de``, ``fr`` ...): a known non-English code must never be re-detected as ``en``."""
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return None
    raw = str(code).strip()
    if not raw or raw.lower() == "auto":
        return None
    if raw in LANGUAGES:
        return raw
    key = raw.lower().replace("_", "-")
    if key in _LANGUAGE_ALIASES:
        return _LANGUAGE_ALIASES[key]
    base = key.split("-", 1)[0]
    if base in ("en", "ja", "ko"):
        return base
    return "other"
