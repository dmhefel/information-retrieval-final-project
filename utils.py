from pathlib import Path
import json, jsonlines
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('msmarco-distilbert-base-v3') #encoder for sbert
#global variable for determing how long the text for chunking and going into sbert encoder
CHUNKSIZE = 300



jane_austen = '''Darcy only smiled; and the general pause which ensued made
      Elizabeth tremble lest her mother should be exposing herself
      again. She longed to speak, but could think of nothing to say;
      and after a short silence Mrs. Bennet began repeating her thanks
      to Mr. Bingley for his kindness to Jane, with an apology for
      troubling him also with Lizzy. Mr. Bingley was unaffectedly civil
      in his answer, and forced his younger sister to be civil also,
      and say what the occasion required. She performed her part indeed
      without much graciousness, but Mrs. Bennet was satisfied, and
      soon afterwards ordered her carriage. Upon this signal, the
      youngest of her daughters put herself forward. The two girls had
      been whispering to each other during the whole visit, and the
      result of it was, that the youngest should tax Mr. Bingley with
      having promised on his first coming into the country to give a
      ball at Netherfield.

      Lydia was a stout, well-grown girl of fifteen, with a fine
      complexion and good-humoured countenance; a favourite with her
      mother, whose affection had brought her into public at an early
      age. She had high animal spirits, and a sort of natural
      self-consequence, which the attention of the officers, to whom
      her uncle’s good dinners, and her own easy manners recommended
      her, had increased into assurance. She was very equal, therefore,
      to address Mr. Bingley on the subject of the ball, and abruptly
      reminded him of his promise; adding, that it would be the most
      shameful thing in the world if he did not keep it. His answer to
      this sudden attack was delightful to their mother’s ear:

      “I am perfectly ready, I assure you, to keep my engagement; and
      when your sister is recovered, you shall, if you please, name the
      very day of the ball. But you would not wish to be dancing when
      she is ill.”

      Lydia declared herself satisfied. “Oh! yes—it would be much
      better to wait till Jane was well, and by that time most likely
      Captain Carter would be at Meryton again. And when you have given
      _your_ ball,” she added, “I shall insist on their giving one
      also. I shall tell Colonel Forster it will be quite a shame if he
      does not.”

      Mrs. Bennet and her daughters then departed, and Elizabeth
      returned instantly to Jane, leaving her own and her relations’
      behaviour to the remarks of the two ladies and Mr. Darcy; the
      latter of whom, however, could not be prevailed on to join in
      their censure of _her_, in spite of all Miss Bingley’s witticisms
      on _fine eyes_.




      Chapter 10

      The day passed much as the day before had done. Mrs. Hurst and
      Miss Bingley had spent some hours of the morning with the
      invalid, who continued, though slowly, to mend; and in the
      evening Elizabeth joined their party in the drawing-room. The
      loo-table, however, did not appear. Mr. Darcy was writing, and
      Miss Bingley, seated near him, was watching the progress of his
      letter and repeatedly calling off his attention by messages to
      his sister. Mr. Hurst and Mr. Bingley were at piquet, and Mrs.
      Hurst was observing their game.

      Elizabeth took up some needlework, and was sufficiently amused in
      attending to what passed between Darcy and his companion. The
      perpetual commendations of the lady, either on his handwriting,
      or on the evenness of his lines, or on the length of his letter,
      with the perfect unconcern with which her praises were received,
      formed a curious dialogue, and was exactly in union with her
      opinion of each.

      “How delighted Miss Darcy will be to receive such a letter!”

      He made no answer.

      “You write uncommonly fast.”

      “You are mistaken. I write rather slowly.”

      “How many letters you must have occasion to write in the course
      of a year! Letters of business, too! How odious I should think
      them!”

      “It is fortunate, then, that they fall to my lot instead of
      yours.”

      “Pray tell your sister that I long to see her.”

      “I have already told her so once, by your desire.”

      “I am afraid you do not like your pen. Let me mend it for you. I
      mend pens remarkably well.”

      “Thank you—but I always mend my own.”

      “How can you contrive to write so even?”

      He was silent.

      “Tell your sister I am delighted to hear of her improvement on
      the harp; and pray let her know that I am quite in raptures with
      her beautiful little design for a table, and I think it
      infinitely superior to Miss Grantley’s.”

      “Will you give me leave to defer your raptures till I write
      again? At present I have not room to do them justice.”

      “Oh! it is of no consequence. I shall see her in January. But do
      you always write such charming long letters to her, Mr. Darcy?”

      “They are generally long; but whether always charming it is not
      for me to determine.”

      “It is a rule with me, that a person who can write a long letter
      with ease, cannot write ill.”
'''

#function that chunks the text
def chunk_text(doc):
    #doc is the string representing the text of document
    tokens = doc.split()
    chunked_tokens = [tokens[i:i + CHUNKSIZE] for i in range(0, len(tokens), CHUNKSIZE)]
    return [" ".join(i) for i in chunked_tokens]


#chunk text
chunked_text = chunk_text(jane_austen)
#get the sbert vectors for chunks
embeddings = model.encode(chunked_text)
# embeddings = []
# for chunk in chunked_text:
#     embeddings.append(model.encode(chunk))
#embeddings = np.arange(2, 11).reshape(3,3) #this gives array of [[2,3,4],[5,6,7],[8,9,10]]
#print(len(embeddings))
#print(embeddings)
#add all embeddings
averaged_embedding = embeddings[0]
#print(averaged_embedding)
if len(embeddings)>1:
    for i in range(1,len(embeddings)):
        averaged_embedding += embeddings[i]
#print(averaged_embedding)
averaged_embedding=averaged_embedding/len(embeddings)
print(averaged_embedding)

#what if doc is empty
empty_str = ""
print("Empty string:")
print(model.encode(empty_str))

#Comparing to doc and original embedding from pa5
doc1 = "Early last week, a postcard advertising a rally for Mitt Romney arrived at the home of Pam Arnold Powers and her husband, Kelly. As undecided voters, the couple had grown accustomed to such invites. They regularly received mail from Rick Perry and Ron Paul, and Romney himself called several times a week, clogging up their voice mail with automated messages that began \u201cPamela, this is Mitt.\u201d \u201cThey use our names!\u201d said Ms. Powers, a gregarious 47 year old who, likewise, considers herself on a first-name basis with Mitt, Newt, Rick and the other Republican hopefuls. The Powerses started concentrating on the Iowa caucuses about a month ago, spending $200 on tickets to a Dec. 10 debate. They have spent days and weeks warming and cooling to candidates. They are trying to recapture the electricity they felt four years ago when Michelle Obama took Ms. Powers\u2019s arm in front of the pork tent at the Iowa State Fair, the first step in their journey off the Republican rolls and into the fold of Obama voters. Now, political disappointment and personal progression have led them back to the GOP, but the Powerses are scrambling to find someone who possesses the attributes on their checklist \u2014 electability, passion, depth and strong moral values. Like the vast ranks of their ambivalent brethren who will determine the winner of the Iowa caucuses and possibly the Republican nominee, the Powerses are still having a tough time. \u201cI\u2019ve got to figure it out by Tuesday,\u201d Ms. Powers said. The Romney rally At 7:10 a.m. Friday, Mr. Powers, a 50-year-old mortgage banker in tapered-temple glasses, arrived for the Romney rally outside a local supermarket in a red golf sweatshirt. His wife, who also thought the event would be held indoors, arrived soon after in a yellow sweater, black vest and paisley-printed slacks. Romney staff members offered them signs to hold. They declined, instead keeping their hands warm in their pockets. Romney\u2019s bus arrived in an area of the parking lot marked off by upended shopping carts. The Powerses cheered as the candidate, his wife and New Jersey Gov. Chris Christie took turns testifying to Mitt Romney\u2019s love of America. As Romney thanked people for coming, Ms. Powers found herself deep in conversation with another undecided voter at the foot of the stage. They talked about Romney\u2019s trouble connecting, Perry\u2019s trouble speaking, Paul\u2019s radicalism, Michele Bachmann\u2019s inexperience and Rick Santorum\u2019s fervor. The Powerses thawed for a few minutes in the supermarket and then drove their beige Saab downtown to give Newt Gingrich, who they had all but written off, a last look at a \u201cMoms Matter Coffee Break\u201d event. The couple found seats in the back of the small room decorated with Elvis album covers. On stage, GOP pollster Frank Luntz warmed up the audience by showing off his stars-and-stripes sneakers (\u201cespecially made for me by K-Swiss\u201d) and asking undecided voters to raise their hands. The Powerses, like most in the room, lifted their arms. Moments later, Gingrich took the stage. As he spoke about how \u201cin a different world it would have been great not to have been divorced,\u201d Ms. Powers slowed the chewing of her gum. She nodded approvingly as Gingrich talked about how \u201crights come from our creator.\u201d When Gingrich made the case that he is more electable than Romney (\u201cI\u2019m a more effective debater\u201d) and decried the millions of dollars spent on negative ads to run his name through the mud, Mr. Powers momentarily stopped checking the photos on his phone to listen. At the end of the event, Gingrich choked up as he spoke about his mother\u2019s death and the impact \u201cthe real problems of real people in my family\u201d had on his policy thinking. Ms. Powers thoughts turned to her own mother, who had died on Christmas Eve 2009, after a battle with cancer. The ordeal, she said, strengthened her Catholic faith, which in turn led her to seek a Republican candidate who shares her moral values. When Gingrich finished speaking, his cheeks shining with tears, she clapped wildly. \u201cPeople are going to go nuts on the crying, but that was as real as anything you can see,\u201d she said as she left the coffee shop. \u201cI thought Newt was off my ballot. But Mitt doesn\u2019t make that connection.\u201d The couple fueled up on moo shoo pork slices at Fong\u2019s Pizza and talked about Santorum as a \u201ca victorious underdog\u201d and Romney as a \u201clast man standing.\u201d Mr. Powers lamented the \u201cbombardment\u201d of anti-Gingrich television ads from Romney-supporting super PACs. \u201cIt has influenced us,\u201d he said unhappily. His wife excused Romney of any blame for the ads and then spotted one of Gingrich\u2019s daughters walking into the restaurant. \u201cYou picked a great place!\u201d she called out to her. Hearing no response, she turned back to the fortune cookies on the table. \u201cOoooh look!\u201d she said, reading the fortune: \u201cLinger over dinner discussions this week for needed advice.\u201d \u2018No one talks like that\u2019 The couple stopped home to check voice mails (Dan Quayle called on behalf of Romney, Romney called on behalf of Romney) and changed into warmer clothes for the drive out to Marshalltown to see Santorum. As their car slid by acres of crushed pale cornstalk, Ms. Powers rebuked her husband for failing to photograph her \u201carm-in-arm with Mitt today\u201d and to be ready with the camera \u201cwhen Rick comes.\u201d Each booth at the Legends American Grill had its own flat-panel television tuned to the Iowa State football game. In a space off the dining room, a small group of potential voters, outnumbered by the national media, sat waiting for the candidate. The Powerses squeezed into the back and craned their necks to see Santorum\u2019s entrance. He wore an Iowa State sweater-vest and spoke under muted TVs tuned to C-SPAN, which showed him speaking, on a seven-second delay, to the voters in the room. Ms. Powers clapped vigorously as Santorum talked about \u201cthe crossroads of American civilization\u201d and asserted that \u201cI know life begins at conception.\u201d As the event plodded on, her husband grew less interested, preferring to text friends as Santorum responded to short questions with long answers. After he spoke for an hour, audible sighs emanated from reporters in the room and a waitress adjusted the thermostat, which read 83 degrees. Santorum, standing before a lighted fireplace, kept talking and, at the one-hour 22-minute mark, he took the last question of the evening. \u201cHow will you conserve your integrity when you\u2019re operating in the Washington machine?\u201d Ms. Powers asked from the back. She maintained eye contact with Santorum throughout his lengthy answer. When the event broke up, she stepped into the dining room with a broad smile across her face. \u201cDone!\u201d she exclaimed, her face flush from the heat. \u201cI know who I\u2019m caucusing for. No one talks like that.\u201d Her husband, less impressed, shrugged. On the drive back to Des Moines, night blotted out the barren fields, and the couple debated Santorum\u2019s performance. \u201cHis answers are too long,\u201d said Mr. Powers. \u201cThat\u2019s the beauty of Rick Santorum,\u201d his wife countered. They talked about his answer on Iran. \u201cHe said we\u2019re going to blow \u2019em up,\u201d said Mr. Powers. \u201cI didn\u2019t hear that at all,\u201d Ms. Powers said. They compared his substantive answers to Romney\u2019s patriotic \u201cpep rally,\u201d but then Ms. Powers suggested she hadn\u2019t decided after all. \u201cIt would be interesting, at the very least,\u201d Ms. Powers said, \u201cto go see Mitt in that kind of venue.\u201d"
chunked_text = chunk_text(doc1)
embeddings = model.encode(chunked_text)
averaged_embedding = embeddings[0]
if len(embeddings)>1:
    for i in range(1,len(embeddings)):
        averaged_embedding += embeddings[i]
#print(averaged_embedding)
averaged_embedding=averaged_embedding/len(embeddings)
print("Averaged embedding:",averaged_embedding)
original_embedding = [-0.1003614217042923, -0.17592774331569672, -0.0013650379842147231, 0.3221243917942047, -0.4316238462924957, -0.08010592311620712, 0.24913814663887024, 0.005946469493210316, 0.07465922832489014, 0.05768083781003952, 0.39409005641937256, -0.30903494358062744, 0.06167539581656456, -0.22649888694286346, -0.5325678586959839, -0.275812566280365, -0.7214683294296265, 0.12269286811351776, -0.5530353784561157, -0.0882389172911644, 0.36213958263397217, -0.08717823773622513, -0.033956486731767654, 0.04080967977643013, 0.004154857713729143, -0.05979607626795769, 0.39253655076026917, -0.6382433176040649, 0.006670936476439238, -0.19655638933181763, -0.16407881677150726, 0.24121515452861786, -0.04777102917432785, 0.6036078333854675, 0.301099956035614, 0.1741647571325302, 0.26164767146110535, -0.11365540325641632, 0.34773576259613037, -0.00632130540907383, 0.4193155765533447, -0.09748804569244385, 0.2956778109073639, -0.026598701253533363, -0.07409979403018951, -0.17151537537574768, 0.4026145040988922, -0.02254166454076767, 0.033443383872509, -0.268889456987381, -0.36734047532081604, -0.1301838457584381, 0.13150039315223694, 0.6813964247703552, 0.2813259959220886, -0.0010712072253227234, -0.09857466071844101, -0.05631634593009949, -0.05121512711048126, -0.21942612528800964, -0.1837892383337021, -0.32266414165496826, -0.01926170103251934, 0.3338237404823303, -0.2805026173591614, -0.007413278799504042, 0.31918615102767944, 0.17679041624069214, 0.34409767389297485, -0.09786713868379593, -0.03160921484231949, -0.9653500914573669, 0.4114649295806885, 0.33368217945098877, 0.3117489814758301, 0.25534096360206604, 0.13249072432518005, 0.05989322066307068, -0.15126696228981018, -0.30028805136680603, -0.1496601402759552, 0.4368021786212921, -0.38818904757499695, 0.5736650228500366, -0.036080941557884216, -0.19760742783546448, 0.5029618144035339, -0.08493272215127945, 0.1926271915435791, 0.1892312914133072, -0.6797239780426025, -0.062431856989860535, -0.10464207828044891, 0.2987561821937561, -0.43817174434661865, -0.5104189515113831, 0.19692805409431458, -0.5169658660888672, 0.2922484576702118, 0.180128812789917, 0.21373996138572693, -0.5593767166137695, -0.1473350077867508, -0.1296018362045288, -0.218180850148201, 0.19835174083709717, 0.16437619924545288, -0.45810970664024353, -0.25360071659088135, 0.15734261274337769, -0.04916058108210564, 0.1982298344373703, -0.03279377147555351, -0.49621424078941345, 0.004981345031410456, 0.3399502635002136, 0.28654852509498596, -0.6353287100791931, -0.21562328934669495, 0.051874447613954544, 0.2965974807739258, 0.20201267302036285, 0.23890073597431183, -0.21344374120235443, -0.006015460472553968, 0.11131761968135834, -0.21718277037143707, 0.5687927603721619, -0.056639980524778366, 0.17366856336593628, 0.08359786123037338, -0.5661779642105103, 0.6572712063789368, -0.2813144326210022, 0.2344980537891388, 0.4228730797767639, 0.16795577108860016, -0.20875504612922668, 0.11186263710260391, 0.2586379051208496, 0.4881605803966522, 0.2417357861995697, 0.32433009147644043, 0.30791106820106506, 0.5972915291786194, -0.048853982239961624, 0.4433554410934448, 0.09560878574848175, 0.16059993207454681, 0.2651435434818268, 0.2604679763317108, -0.3257084786891937, 0.17035476863384247, 0.4445986747741699, -0.15330111980438232, -0.12026941776275635, 0.100617416203022, 0.14143253862857819, 0.6845951080322266, 0.31035593152046204, 0.44695112109184265, -0.7387138605117798, -0.35734066367149353, 0.11838424950838089, -0.26170870661735535, 0.7990721464157104, -0.29665541648864746, 0.13649892807006836, 0.41864266991615295, -0.14579135179519653, -0.19513992965221405, -0.11348024010658264, 0.1737954318523407, 0.029823649674654007, 0.545930027961731, 0.2545402944087982, -0.15902604162693024, -0.3476860523223877, 0.009988955222070217, -0.18575622141361237, 0.10484781861305237, 0.010856179520487785, -0.18187305331230164, 0.2918866276741028, -0.1411004513502121, -0.1147804856300354, -0.2604365050792694, -0.19055025279521942, 0.11381439119577408, 0.13550463318824768, 0.020616739988327026, -0.19034507870674133, -0.339741975069046, -0.4270031452178955, 0.3296704888343811, 0.04939538612961769, 0.012233272194862366, -0.2823036313056946, 0.4503299295902252, 0.226493239402771, -0.004800786264240742, -0.3461427688598633, 0.27084681391716003, -0.27406975626945496, -0.2266913801431656, 0.4763481914997101, 0.5183078050613403, 0.23314858973026276, -0.08151064068078995, 0.2637520730495453, 0.1276090443134308, 0.47120893001556396, 0.04912711679935455, 0.18726672232151031, -0.2899673283100128, 0.4769763946533203, 0.21914008259773254, -0.29030025005340576, -0.26043808460235596, 0.023488519713282585, -0.1494854837656021, 0.48282739520072937, 0.1466660499572754, 0.011469289660453796, -0.19849000871181488, 0.46141576766967773, 0.05555110424757004, 0.02821470983326435, -0.606643557548523, 0.33415549993515015, -0.17578795552253723, 0.08566411584615707, 0.16103926301002502, -0.22618013620376587, 0.49919289350509644, -0.7437891364097595, 0.001696338877081871, -0.10082580149173737, 0.3641955554485321, -0.3051324188709259, 0.055642638355493546, 0.08113111555576324, 0.30642765760421753, -0.14388476312160492, -0.31681403517723083, 0.07354632765054703, 0.2736720144748688, 0.1826942265033722, 0.5217465162277222, -0.5088763236999512, 0.5383721590042114, 0.03317764773964882, 0.026038799434900284, -0.022302579134702682, -0.3085317611694336, 0.2807803153991699, 0.2324250191450119, 0.227766215801239, 0.404811829328537, -0.06454495340585709, -0.4406530559062958, -0.23425383865833282, -0.5074904561042786, 0.47164300084114075, 0.07394569367170334, -0.6039252877235413, 0.29117122292518616, -0.18013006448745728, -0.13146965205669403, -0.3833279311656952, 0.4362892806529999, -0.013674171641469002, 0.12400629371404648, -0.04329248145222664, 0.46693500876426697, 0.16260656714439392, -0.4005238711833954, 0.6586775183677673, -0.059415727853775024, -0.0765116959810257, -0.09383241087198257, -0.12018191814422607, -0.13477648794651031, -0.15449556708335876, -0.07945317029953003, -0.14790700376033783, 0.20376574993133545, -0.2025051862001419, -0.28832438588142395, 0.0013122885720804334, 0.7229810357093811, 0.4550640285015106, 0.5945533514022827, 0.47313088178634644, -0.0124121755361557, 0.6280959248542786, 0.10823317617177963, 0.5078432559967041, -0.5068715810775757, 0.23006092011928558, 0.5747027397155762, -0.209723562002182, -0.1866961270570755, 0.2538772523403168, -0.2753077447414398, -0.4098037779331207, -0.27646011114120483, -1.0557414293289185, -1.18607759475708, -0.3182431161403656, -0.41785407066345215, 0.12053409218788147, 0.399593323469162, 0.6697329878807068, 0.011432417668402195, 0.14845193922519684, 0.1704491674900055, -0.0072845411486923695, 0.5719186663627625, 0.47508421540260315, -0.14979754388332367, -0.17370854318141937, -0.17674724757671356, 0.23016338050365448, 0.3513270914554596, -0.4387279748916626, 0.4206687808036804, -0.396486759185791, 0.2869845926761627, -0.22819791734218597, -0.017424315214157104, -0.213575541973114, 0.21820317208766937, -0.31827062368392944, 0.3192631006240845, -0.45371687412261963, 0.26594915986061096, 0.03194837272167206, 0.10794486850500107, -0.13673116266727448, 0.23508019745349884, 0.3045583963394165, 0.35192784667015076, 0.469395250082016, 0.5248669385910034, -0.3086974024772644, -0.2672678828239441, -0.2420206516981125, 0.0561218336224556, -0.12511780858039856, 0.1370116025209427, -0.19703975319862366, 0.06465283781290054, 0.6530288457870483, 0.06038516014814377, -0.11697530001401901, 0.08726296573877335, -0.3838491439819336, -0.5735805034637451, -0.014179055579006672, 0.22206991910934448, 0.3035818040370941, 0.14241787791252136, -0.3540567457675934, 0.3223090171813965, 0.18403710424900055, 0.208377867937088, 0.08780836313962936, -0.1889897733926773, 0.14141052961349487, 0.031003586947917938, 0.024814879521727562, 0.09738396853208542, -0.41463664174079895, -0.02749720960855484, -0.05134487524628639, -0.1030362918972969, -0.07958973944187164, -0.22243209183216095, -0.3213939666748047, -0.32337483763694763, -0.2696000635623932, 0.45783039927482605, -4.218793401378207e-05, 0.44463980197906494, -0.08634399622678757, -0.1489039957523346, 0.12058219313621521, -0.22554081678390503, -0.09895040094852448, -0.2149747610092163, -0.32877713441848755, 0.6174685955047607, 0.18449771404266357, 0.04045981541275978, 0.08572406321763992, 0.3639489412307739, -0.20181366801261902, -0.2779751718044281, -0.05753689259290695, -0.4873408377170563, 0.083522267639637, 0.40640223026275635, 0.3633364737033844, 0.28953588008880615, -0.10077864676713943, 0.11923190951347351, -0.00032372193527407944, -0.4120471477508545, 0.3758368492126465, -0.0999157652258873, -0.48696497082710266, -0.0450943298637867, 0.4423685669898987, -0.44434285163879395, -0.642379105091095, -0.08425626158714294, 0.049145881086587906, -0.37339872121810913, -0.8301497101783752, 0.06775858253240585, -0.5221571922302246, -0.126400887966156, 0.6315815448760986, 0.5001962780952454, 0.38635340332984924, -0.34933793544769287, 0.6689406037330627, -0.034864723682403564, 0.13060182332992554, -0.3322158455848694, -0.32662785053253174, -0.6635230779647827, -0.15616698563098907, -0.17133648693561554, -0.11763603240251541, -0.026756634935736656, 0.07767655700445175, -0.029285691678524017, -0.31115007400512695, -0.019486596807837486, 0.5276125073432922, 0.1273486614227295, 0.3017347455024719, 0.3629927635192871, -0.33034056425094604, -0.3308189809322357, 0.2756892144680023, -0.28492042422294617, 0.14746126532554626, -0.3608209788799286, -0.14612863957881927, -0.038631897419691086, -0.3302665054798126, -0.018506648018956184, -0.04046424478292465, -0.07204870879650116, 0.4082343578338623, -0.011002616956830025, 0.082119882106781, 0.14282192289829254, -0.03516067937016487, 0.0965990498661995, -0.08656420558691025, 0.19887492060661316, -0.34659454226493835, -0.10113657265901566, 0.36661437153816223, -0.7374969720840454, -0.15534362196922302, 0.13647308945655823, 0.13511402904987335, 0.5083003640174866, -0.015747781842947006, -0.07192511856555939, -0.4003397524356842, -0.21085241436958313, 0.015471224673092365, 0.09954962134361267, 0.4201085567474365, 0.00865047238767147, 0.6918692588806152, 0.3915262520313263, 0.09229009598493576, -0.0023435275070369244, 0.23658162355422974, -0.057004522532224655, -0.39363834261894226, 0.13200269639492035, -0.2173052877187729, -0.3158387541770935, 0.0248293187469244, 0.010688796639442444, -0.0787992849946022, -0.0748288556933403, 0.17738398909568787, -0.3728990852832794, 0.11715031415224075, -0.2314375340938568, 0.024650149047374725, -0.10127340257167816, -0.010544023476541042, -0.28566738963127136, 0.6055153012275696, 0.16543088853359222, -0.8147707581520081, -0.23605161905288696, -0.317826509475708, -0.07773114740848541, -0.34417036175727844, 0.35417360067367554, 0.10487906634807587, -0.27660664916038513, -0.1840715855360031, -0.7750990986824036, 0.23314984142780304, 0.575684666633606, 0.19590634107589722, 0.5167991518974304, 0.45830902457237244, 0.6609422564506531, -0.11352843791246414, -0.041114576160907745, 0.06281471997499466, -0.16461603343486786, 0.10931858420372009, -0.3167511522769928, 0.17988285422325134, -0.6453870534896851, -0.4050285220146179, 0.14124859869480133, -0.47084590792655945, 0.2188403159379959, -0.43829065561294556, 0.2872870862483978, -0.10754814743995667, 0.12463484704494476, -0.45256713032722473, -0.1234397366642952, -0.6000807285308838, 0.2668936848640442, -0.1511746346950531, -0.166622593998909, 0.20803898572921753, -0.11533400416374207, 0.5299314260482788, -0.1833299845457077, 0.4037327766418457, 0.37060272693634033, 0.2754998207092285, -0.03025384247303009, 0.295787513256073, 0.03203234821557999, 0.11347997188568115, -0.15720252692699432, 0.09437362849712372, -0.04927109554409981, 0.04020159691572189, -0.339938759803772, 0.7708072662353516, 0.034948475658893585, 0.043109770864248276, -0.010522738099098206, -0.34192001819610596, -0.10464439541101456, -0.2044525146484375, 0.039465706795454025, -0.34916847944259644, -0.5581347346305847, 0.022872965782880783, 0.2298920750617981, 0.17186366021633148, -0.8028326034545898, -0.2959096133708954, 0.30377262830734253, 0.07288909703493118, -0.27446919679641724, -0.08713794499635696, -0.2936633229255676, -0.030826538801193237, -0.38231727480888367, -0.5243058800697327, -0.3306310176849365, -0.12616154551506042, 0.061105504631996155, 0.27614086866378784, -0.24378566443920135, 0.3381636142730713, 0.03324265778064728, 0.42601075768470764, -0.08558490872383118, -0.17006799578666687, -0.23237454891204834, -0.11475656926631927, 0.05566098168492317, -0.2146136313676834, 0.3338545262813568, -0.5568385720252991, 0.09872125089168549, 0.039823561906814575, -0.2884521484375, -0.15400578081607819, -0.30301883816719055, 0.16365577280521393, 0.2199181169271469, -0.23834842443466187, -0.8564706444740295, -0.3868709206581116, -0.23996838927268982, 0.693278968334198, -0.3769437372684479, 0.13609810173511505, -0.9531723856925964, 0.1295423060655594, 0.1233365386724472, -0.17976367473602295, 0.07188474386930466, 0.15859736502170563, -0.5002496242523193, -0.3703698217868805, 0.2481217086315155, 0.38330531120300293, 0.357137531042099, -0.07289841026067734, 0.24630025029182434, -0.030012190341949463, -0.6391294598579407, -0.31484749913215637, 0.03947175666689873, -0.10311584174633026, -0.06309042125940323, -0.12933741509914398, -0.3183748126029968, 0.63360995054245, -0.09441045671701431, -0.15853440761566162, -0.23452670872211456, -0.5135942101478577, 0.3722826838493347, 0.2540639340877533, -0.3052162826061249, -0.09535142034292221, 0.2826795279979706, -0.30332037806510925, 0.08935300260782242, 0.19778317213058472, 0.17565424740314484, -0.0354016050696373, -0.504841685295105, 0.2215578258037567, 0.44679734110832214, 0.45307111740112305, -0.14453837275505066, 0.2680635154247284, -0.09026504307985306, 0.5051705837249756, -0.3043968081474304, -0.2875113785266876, 0.14089997112751007, -0.11767446249723434, -0.3046397566795349, -0.2635710835456848, 0.42676588892936707, -0.5254998803138733, -0.22908148169517517, 0.042643483728170395, 0.5144325494766235, -0.1027441918849945, 0.196012482047081, -0.5814242362976074, -0.30726131796836853, -0.00032592006027698517, -0.10192405432462692, -0.3903757333755493, 0.3580731451511383, 0.09412853419780731, -0.25290629267692566, -0.710893452167511, 0.0013568513095378876, -0.19455276429653168, 0.2836415469646454, 0.04308132454752922, -0.39828765392303467, -0.050098687410354614, 0.06534961611032486, -0.7420041561126709, -0.08085114508867264, 0.32843726873397827, 0.11147206276655197, -0.6928338408470154, -0.09503055363893509, 0.13305403292179108, 0.7415687441825867, -0.28660815954208374, 0.36998334527015686, 0.1940782070159912, -0.5057114362716675, -0.032604899257421494, 0.37684133648872375, -0.2803650498390198, -0.591361403465271, 0.08047577738761902, -0.34759896993637085, -0.17098428308963776, -0.6316210031509399, -0.05149095132946968, 0.23700208961963654, 0.31818732619285583, 0.39958322048187256, 0.02380911260843277, -0.00016259726544376463, 0.17509756982326508, 0.1851027011871338, -0.23493769764900208, 0.2289867252111435, -0.5543287992477417, -0.36240634322166443, -0.356930136680603, -0.5329935550689697, -0.0010643847053870559, -0.028850920498371124, -0.5382895469665527, 0.11147947609424591, -0.090720996260643, 0.11604032665491104, 0.45622649788856506, 0.8640995025634766, 0.369075745344162, -0.11394671350717545, 0.28244540095329285, -0.27175405621528625, 0.6965973377227783, 0.2603771686553955, 0.8091067671775818, 0.09110701829195023, 0.41439059376716614, 0.2093130350112915, -0.07366980612277985, -0.007779484149068594, 0.3304611146450043, 0.21815268695354462, 0.17130525410175323, 0.21867266297340393, 0.10673320293426514, 0.12186912447214127, 0.2885500490665436, 0.002644791267812252, 0.03725175932049751, 0.541334331035614, 0.14651770889759064, 0.4106711447238922, 0.19515323638916016, -0.5409489870071411, -0.3036448061466217, 0.03413054719567299, -0.41423749923706055, 0.17752385139465332, 0.41548240184783936, -0.6410931944847107, 0.0366213358938694, -0.1510828137397766, 0.04947635531425476, 0.4077630639076233, -0.0852220356464386, 0.3455185294151306, -0.08358471095561981, 0.03459097072482109, 0.16860243678092957]
original_embedding = np.asarray(original_embedding)
averaged_embedding = averaged_embedding-original_embedding
print("Averaged - original embedding: ", averaged_embedding)

def load_wapo(wapo_jl_path):
    """
    It should be similar to the load_wapo in HW3 with two changes:
    - for each yielded document dict, use "doc_id" instead of "id" as the key to store the document id.
    - convert the value of "published_date" to a readable format e.g. 2021/3/15. You may consider using python datatime package to do this.
    """
    # open the jsonline file
    with open(wapo_jl_path) as f:
        # get line
        line = f.readline()
        # id number
        id = 0
        # while line not empty
        while line:
            # load line as json
            article = json.loads(line)
            # process content by getting each sanitized html sentence and adding to list.
            # content = []
            # for sentence in article['contents']:
            #     if sentence is not None and 'content' in sentence and sentence['type'] == 'sanitized_html':
            #         content.append(sentence['content'])
            # # turn list of sentences into one string
            # content = ' '.join(content)
            # # use regex to get rid of html elements
            # content = re.sub('<[^<]+?>', '', content)
            #process date
            #dateinfo = dt.fromtimestamp(article['published_date'] / 1000)
            #date = str(dateinfo.year) + "/" + str(dateinfo.month) + "/" + str(dateinfo.day)

            date = article['published_date']
            # create dict for the article
            articledict = {'id': id,
                           'title': article["title"] if article['title'] is not None else '',
                           "author": article["author"] if article['author'] is not None else '',
                           "published_date": date,
                           "content_str": article['content_str'],
                           "annotation": article['annotation'],
                           "sbert_vector": article['sbert_vector'], #do we need?
                           #"ft_vector": article['ft_vector'], #do we need?
                           }
            # increment id #
            id += 1
            # get next line
            line = f.readline()
            # yield the dictionary (generator object)
            yield articledict
with jsonlines.open('data/with_avg_sbert', mode='a') as writer:

    data_dir = Path("data")
    wapo_path = data_dir.joinpath('subset_wapo_50k_sbert_ft_filtered.jl')
    wapo = load_wapo(wapo_path)
    article = next(wapo, '')
    while article:
        chunked_text = chunk_text(doc1)
        embeddings = model.encode(chunked_text)
        averaged_embedding = embeddings[0]
        if len(embeddings) > 1:
            for i in range(1, len(embeddings)):
                averaged_embedding += embeddings[i]
        # print(averaged_embedding)
        averaged_embedding = averaged_embedding / len(embeddings)
        article['new_sbert'] = averaged_embedding.tolist()
        writer.write(article)
        print(article['title'])
        article = next(wapo, '')

# data_dir = Path("data")
# wapo_path = data_dir.joinpath("50k.jl")
# documents = [doc['content_str'] for doc in load_wapo(wapo_path)]
#
# print(documents[0])
#
#
#
#
#
#
#
#
#
#
#
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument
# from nltk.tokenize import word_tokenize
#
# data = ["I love machine learning. It's awesome.",
#         "I love coding in python",
#         "I love building chatbots",
#         "they chat amagingly well"]
#
# data = documents
#
# tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
#
# #print(tagged_data)
#
# max_epochs = 10
# vec_size = 50
# alpha = 0.025
#
# model = Doc2Vec(vector_size=vec_size,
#                 alpha=alpha,
#                 min_alpha=0.00025,
#                 min_count=1,
#                 dm=1)
#
# model.build_vocab(tagged_data)
#
# for epoch in range(max_epochs):
#     print('iteration {0}'.format(epoch))
#     model.train(tagged_data,
#                 total_examples=model.corpus_count,
#                 epochs=max_epochs)
#     # decrease the learning rate
#     model.alpha -= 0.0002
#     # fix the learning rate, no decay
#     model.min_alpha = model.alpha
#
# model.save("d2v.model")
# print("Model Saved")
#
#
#
# from gensim.models.doc2vec import Doc2Vec
#
# model= Doc2Vec.load("d2v.model")
# #to find the vector of a document which is not in training data
# test_data = word_tokenize("inventions scientific discovery".lower())
# v1 = model.infer_vector(test_data)
# print("V1_infer", v1)
#
# # to find most similar doc using tags
# similar_doc = model.dv.most_similar('1')
# print(similar_doc)
#
#
# # to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.dv['1'])
