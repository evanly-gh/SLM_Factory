I want to run a small language model on my Android phone for different things, without the cloud. Below are the options I have and 60 things I want to do, each on one of my phones. For each one, tell me which options would work well enough on that phone, which wouldn't and why, and which one I should use.

**Phones:** Galaxy S23, OnePlus 15, OnePlus 8 Pro, Redmi 9A.

**Models:** SmolLM2 (Hugging Face), Gemma 3 (Google), Qwen3 and Qwen3.5 (Alibaba). All of them are fine-tuned for whatever the request is.

**Choices.** An option is a model plus three choices:

- **Precision:** how the model's weights are stored. 4-bit (Q4) is smallest and fastest and loses a little quality; 8-bit (Q8) is in between; F16 is full quality and largest.
- **Runtime:** the inference engine that runs the model on the phone, llama.cpp or MNN.
- **Backend:** whether the runtime runs the model on the phone's CPU or its GPU.

| Option | Model | Precision | Runtime | Backend |
|---|---|---|---|---|
| C001 | SmolLM2-135M | Q4_0 | llama.cpp | CPU |
| C002 | SmolLM2-135M | Q4_0 | llama.cpp | GPU |
| C003 | SmolLM2-135M | Q4 | MNN | CPU |
| C004 | SmolLM2-135M | Q4 | MNN | GPU |
| C005 | SmolLM2-135M | Q8_0 | llama.cpp | CPU |
| C006 | SmolLM2-135M | Q8_0 | llama.cpp | GPU |
| C007 | SmolLM2-135M | Q8 | MNN | CPU |
| C008 | SmolLM2-135M | Q8 | MNN | GPU |
| C009 | SmolLM2-135M | F16 | llama.cpp | CPU |
| C010 | SmolLM2-135M | F16 | llama.cpp | GPU |
| C011 | SmolLM2-135M | F16 | MNN | CPU |
| C012 | SmolLM2-135M | F16 | MNN | GPU |
| C013 | Gemma3-270M | Q4_0 | llama.cpp | CPU |
| C014 | Gemma3-270M | Q4_0 | llama.cpp | GPU |
| C015 | Gemma3-270M | Q4 | MNN | CPU |
| C016 | Gemma3-270M | Q4 | MNN | GPU |
| C017 | Gemma3-270M | Q8_0 | llama.cpp | CPU |
| C018 | Gemma3-270M | Q8_0 | llama.cpp | GPU |
| C019 | Gemma3-270M | Q8 | MNN | CPU |
| C020 | Gemma3-270M | Q8 | MNN | GPU |
| C021 | Gemma3-270M | F16 | llama.cpp | CPU |
| C022 | Gemma3-270M | F16 | llama.cpp | GPU |
| C023 | Gemma3-270M | F16 | MNN | CPU |
| C024 | Gemma3-270M | F16 | MNN | GPU |
| C025 | SmolLM2-360M | Q4_0 | llama.cpp | CPU |
| C026 | SmolLM2-360M | Q4_0 | llama.cpp | GPU |
| C027 | SmolLM2-360M | Q4 | MNN | CPU |
| C028 | SmolLM2-360M | Q4 | MNN | GPU |
| C029 | SmolLM2-360M | Q8_0 | llama.cpp | CPU |
| C030 | SmolLM2-360M | Q8_0 | llama.cpp | GPU |
| C031 | SmolLM2-360M | Q8 | MNN | CPU |
| C032 | SmolLM2-360M | Q8 | MNN | GPU |
| C033 | SmolLM2-360M | F16 | llama.cpp | CPU |
| C034 | SmolLM2-360M | F16 | llama.cpp | GPU |
| C035 | SmolLM2-360M | F16 | MNN | CPU |
| C036 | SmolLM2-360M | F16 | MNN | GPU |
| C037 | Qwen3-0.6B | Q4_0 | llama.cpp | CPU |
| C038 | Qwen3-0.6B | Q4_0 | llama.cpp | GPU |
| C039 | Qwen3-0.6B | Q4 | MNN | CPU |
| C040 | Qwen3-0.6B | Q4 | MNN | GPU |
| C041 | Qwen3-0.6B | Q8_0 | llama.cpp | CPU |
| C042 | Qwen3-0.6B | Q8_0 | llama.cpp | GPU |
| C043 | Qwen3-0.6B | Q8 | MNN | CPU |
| C044 | Qwen3-0.6B | Q8 | MNN | GPU |
| C045 | Qwen3-0.6B | F16 | llama.cpp | CPU |
| C046 | Qwen3-0.6B | F16 | llama.cpp | GPU |
| C047 | Qwen3-0.6B | F16 | MNN | CPU |
| C048 | Qwen3-0.6B | F16 | MNN | GPU |
| C049 | Qwen3.5-0.8B | Q4_0 | llama.cpp | CPU |
| C050 | Qwen3.5-0.8B | Q4_0 | llama.cpp | GPU |
| C051 | Qwen3.5-0.8B | Q4 | MNN | CPU |
| C052 | Qwen3.5-0.8B | Q4 | MNN | GPU |
| C053 | Qwen3.5-0.8B | Q8_0 | llama.cpp | CPU |
| C054 | Qwen3.5-0.8B | Q8_0 | llama.cpp | GPU |
| C055 | Qwen3.5-0.8B | Q8 | MNN | CPU |
| C056 | Qwen3.5-0.8B | Q8 | MNN | GPU |
| C057 | Qwen3.5-0.8B | F16 | llama.cpp | CPU |
| C058 | Qwen3.5-0.8B | F16 | llama.cpp | GPU |
| C059 | Qwen3.5-0.8B | F16 | MNN | CPU |
| C060 | Qwen3.5-0.8B | F16 | MNN | GPU |
| C061 | Qwen3-1.7B | Q4_0 | llama.cpp | CPU |
| C062 | Qwen3-1.7B | Q4_0 | llama.cpp | GPU |
| C063 | Qwen3-1.7B | Q4 | MNN | CPU |
| C064 | Qwen3-1.7B | Q4 | MNN | GPU |
| C065 | Qwen3-1.7B | Q8_0 | llama.cpp | CPU |
| C066 | Qwen3-1.7B | Q8_0 | llama.cpp | GPU |
| C067 | Qwen3-1.7B | Q8 | MNN | CPU |
| C068 | Qwen3-1.7B | Q8 | MNN | GPU |
| C069 | Qwen3-1.7B | F16 | llama.cpp | CPU |
| C070 | Qwen3-1.7B | F16 | llama.cpp | GPU |
| C071 | Qwen3-1.7B | F16 | MNN | CPU |
| C072 | Qwen3-1.7B | F16 | MNN | GPU |
| C073 | Qwen3.5-2B | Q4_0 | llama.cpp | CPU |
| C074 | Qwen3.5-2B | Q4_0 | llama.cpp | GPU |
| C075 | Qwen3.5-2B | Q4 | MNN | CPU |
| C076 | Qwen3.5-2B | Q4 | MNN | GPU |
| C077 | Qwen3.5-2B | Q8_0 | llama.cpp | CPU |
| C078 | Qwen3.5-2B | Q8_0 | llama.cpp | GPU |
| C079 | Qwen3.5-2B | Q8 | MNN | CPU |
| C080 | Qwen3.5-2B | Q8 | MNN | GPU |
| C081 | Qwen3.5-2B | F16 | llama.cpp | CPU |
| C082 | Qwen3.5-2B | F16 | llama.cpp | GPU |
| C083 | Qwen3.5-2B | F16 | MNN | CPU |
| C084 | Qwen3.5-2B | F16 | MNN | GPU |
| C085 | Qwen3-4B | Q4_0 | llama.cpp | CPU |
| C086 | Qwen3-4B | Q4_0 | llama.cpp | GPU |
| C087 | Qwen3-4B | Q4 | MNN | CPU |
| C088 | Qwen3-4B | Q4 | MNN | GPU |
| C089 | Qwen3-4B | Q8_0 | llama.cpp | CPU |
| C090 | Qwen3-4B | Q8_0 | llama.cpp | GPU |
| C091 | Qwen3-4B | Q8 | MNN | CPU |
| C092 | Qwen3-4B | Q8 | MNN | GPU |
| C093 | Qwen3-4B | F16 | llama.cpp | CPU |
| C094 | Qwen3-4B | F16 | llama.cpp | GPU |
| C095 | Qwen3-4B | F16 | MNN | CPU |
| C096 | Qwen3-4B | F16 | MNN | GPU |
| C097 | Qwen3.5-4B | Q4_0 | llama.cpp | CPU |
| C098 | Qwen3.5-4B | Q4_0 | llama.cpp | GPU |
| C099 | Qwen3.5-4B | Q4 | MNN | CPU |
| C100 | Qwen3.5-4B | Q4 | MNN | GPU |
| C101 | Qwen3.5-4B | Q8_0 | llama.cpp | CPU |
| C102 | Qwen3.5-4B | Q8_0 | llama.cpp | GPU |
| C103 | Qwen3.5-4B | Q8 | MNN | CPU |
| C104 | Qwen3.5-4B | Q8 | MNN | GPU |

**What "works" means.** An option works for a request if it fits all four:

- **M (memory):** it doesn't use more memory than that kind of app can have without the phone closing it.
- **P (prompt speed):** it reads my input fast enough to start answering in time.
- **D (generation speed):** it writes the answer fast enough.
- **E (battery):** it doesn't use too much battery for how often I use it.

Some options can't run on a phone at all (they don't fit, fail to load or crash, or the GPU backend doesn't work there).

**Which to use:** the most capable option that works: the biggest model, then the highest precision; if several runtimes or backends tie, the faster one, then the one that uses less battery.

## What I want

- **S01** · OnePlus 8 Pro: I want to tell my voice assistant things like "lunch with Sam Friday at 1" and have it add the event to my calendar. It should confirm within a couple of seconds; I'd do this a few times a day.
- **S02** · OnePlus 15: I want a shortcuts app where I type what I want and it runs the right action on my phone. I'd use it about twenty times a day and can wait a few seconds.
- **S03** · OnePlus 8 Pro: I want a "fix grammar" button in my email app that corrects a sentence within a few seconds of tapping it.
- **S04** · OnePlus 15: I want my voice assistant to do things when I ask: send a text, set an alarm, check the weather. It should act within a second or two, a couple of dozen times a day.
- **S05** · Redmi 9A: I want to type reminders in plain English into my reminders app and have them set up within a few seconds.
- **S06** · OnePlus 15: I want an app for reminders and weather that understands what I type, on the phone itself. I'd use it about twenty times a day.
- **S07** · OnePlus 15: I want a private assistant app that understands my typed requests without sending anything to the internet. I'd use it about thirty times a day and can wait a few seconds.
- **S08** · Galaxy S23: I'm learning English and keep a daily journal. I want my phone to correct it overnight while it charges, so the corrections are ready in the morning.
- **S09** · OnePlus 15: After I install it, I want an app to go through my 2,000 photo captions and notes overnight and tag the people, places and things in them.
- **S10** · Galaxy S23: I want my phone to summarize the day's recorded calls overnight while it charges, ready in the morning.
- **S11** · Redmi 9A: I'm learning English and keep a daily journal. I want my phone to correct it overnight while it charges, so the corrections are ready in the morning.
- **S12** · OnePlus 8 Pro: I read about twenty medical research abstracts a day and want an app that highlights the drug and disease names as I read, keeping up with my reading.
- **S13** · Galaxy S23: I want to check my whole text history, about 5,000 messages, for spam overnight while my phone charges. It has to finish by morning without using up much of the charge.
- **S14** · OnePlus 15: I want my keyboard to go over what I typed each day, overnight while the phone charges, and give me a mood summary in the morning.
- **S15** · OnePlus 15: I want to check my whole text history, about 5,000 messages, for spam overnight while my phone charges. It has to finish by morning without using up much of the charge.
- **S16** · Galaxy S23: I want my voice assistant to do things when I ask: send a text, set an alarm, check the weather. It should act within a second or two, a couple of dozen times a day.
- **S17** · OnePlus 15: I want my search app to recognize the people, films and products in what I type, so the results are better. Search has to feel quick; I search dozens of times a day.
- **S18** · OnePlus 15: I want a planning app that turns what I type into calendar events. I plan my week in one go, about ten events, and can wait a few seconds for each.
- **S19** · OnePlus 8 Pro: I want my phone to summarize the day's recorded calls overnight while it charges, ready in the morning.
- **S20** · OnePlus 15: I want an app that proofreads my English essays sentence by sentence; the corrections can trickle in while I read.
- **S21** · Galaxy S23: I want a planning app that turns what I type into calendar events. I plan my week in one go, about ten events, and can wait a few seconds for each.
- **S22** · OnePlus 15: I want an English-learning app that corrects my practice sentences. I do about fifty a day and can wait a few seconds for each.
- **S23** · Redmi 9A: I want my phone to summarize the day's recorded calls overnight while it charges, ready in the morning.
- **S24** · OnePlus 8 Pro: When I highlight a sentence in my reference manager, I want it to tag the drugs and diseases in it within a few seconds. A couple of dozen times a day.
- **S25** · OnePlus 8 Pro: I'm building a social app and want it to warn people before they post a comment that sounds angry. The check should take about a second.
- **S26** · OnePlus 15: I want an app where I paste a suspicious text and it tells me whether it's spam. I'd only use it a few times a day and can wait a few seconds.
- **S27** · OnePlus 15: I read about twenty medical research abstracts a day and want an app that highlights the drug and disease names as I read, keeping up with my reading.
- **S28** · Galaxy S23: I want my phone's dialer to summarize my short recorded calls, so I can read the summary right after hanging up. It should start within a second or two, and I read it as it appears.
- **S29** · OnePlus 15: I want my phone to summarize the day's recorded calls overnight while it charges, ready in the morning.
- **S30** · Redmi 9A: When I long-press text on my screen, I want my phone to spot the names, places and products in it and offer actions. It runs in the background; a few seconds is fine.
- **S31** · Galaxy S23: I want to type events in plain English into my calendar app and have it create them. The event should show up within about three seconds.
- **S32** · OnePlus 15: I want to type things like "turn off the kitchen lights" into my smart-home app and have it control the device within a few seconds. Maybe twenty commands a day.
- **S33** · Galaxy S23: I want a notes app that summarizes a short recorded conversation when I tap summarize; I read it as it appears.
- **S34** · Galaxy S23: I want an app where I paste a suspicious text and it tells me whether it's spam. I'd only use it a few times a day and can wait a few seconds.
- **S35** · OnePlus 8 Pro: Every night while my phone charges, I want it to go through the abstracts of the hundred or so papers I saved or got alerts for, and index the drugs and diseases so I can search them. It has to finish by morning.
- **S36** · OnePlus 15: I want my keyboard to tell me the tone of a message before I send it. It has to be instant and can't slow my keyboard down; I send about a hundred messages a day.
- **S37** · OnePlus 15: I want my texting app to check every incoming text for spam before it notifies me, running quietly in the background. I get dozens of texts a day, so it can't eat my battery.
- **S38** · OnePlus 15: I'm learning English and want my keyboard to suggest a grammar fix as soon as I finish each sentence. It can't slow down my typing; I type a couple of hundred sentences a day.
- **S39** · Galaxy S23: I want my phone's voice assistant to work out what I'm asking for and hand it to the right app, all on the phone. I use it maybe twenty times a day, and it has to answer within a second or two of me finishing talking.
- **S40** · OnePlus 15: I'm building my bank's app and want its help chat to work out what a customer is asking about (a transfer, their card, their balance) on the phone itself. Customers expect a reply within a couple of seconds, but it's only used once in a while.
- **S41** · OnePlus 15: When I open a conversation with unread messages, I want my messaging app to summarize what I missed, appearing as I read.
- **S42** · Redmi 9A: I want to say reminder and weather commands to my watch and have my phone work them out in the background.
- **S43** · OnePlus 15: I save about five articles a day to read later and want an app to tag the people, places and products in them as I read.
- **S44** · OnePlus 15: I want my journaling app to tag the emotions in my entry after I save it. I write one entry a day and can wait a few seconds.
- **S45** · Galaxy S23: I want an app for reminders and weather that understands what I type, on the phone itself. I'd use it about twenty times a day.
- **S46** · Galaxy S23: I want my keyboard to go over what I typed each day, overnight while the phone charges, and give me a mood summary in the morning.
- **S47** · OnePlus 15: I want a notes app that summarizes a short recorded conversation when I tap summarize; I read it as it appears.
- **S48** · OnePlus 8 Pro: I want my keyboard to go over what I typed each day, overnight while the phone charges, and give me a mood summary in the morning.
- **S49** · OnePlus 8 Pro: After I install it, I want an app to go through my 2,000 photo captions and notes overnight and tag the people, places and things in them.
- **S50** · Redmi 9A: I want my keyboard to go over what I typed each day, overnight while the phone charges, and give me a mood summary in the morning.
- **S51** · OnePlus 8 Pro: I want my texting app to check every incoming text for spam before it notifies me, running quietly in the background. I get dozens of texts a day, so it can't eat my battery.
- **S52** · OnePlus 15: Every night while my phone charges, I want it to go through the abstracts of the hundred or so papers I saved or got alerts for, and index the drugs and diseases so I can search them. It has to finish by morning.
- **S53** · Redmi 9A: I want to speak requests into my watch and have my phone carry them out in the background. A few seconds is fine, about twenty times a day.
- **S54** · Redmi 9A: I want an English-learning app that corrects my practice sentences. I do about fifty a day and can wait a few seconds for each.
- **S55** · OnePlus 8 Pro: I'm learning English and keep a daily journal. I want my phone to correct it overnight while it charges, so the corrections are ready in the morning.
- **S56** · OnePlus 8 Pro: I want an app for reminders and weather that understands what I type, on the phone itself. I'd use it about twenty times a day.
- **S57** · OnePlus 8 Pro: When I long-press text on my screen, I want my phone to spot the names, places and products in it and offer actions. It runs in the background; a few seconds is fine.
- **S58** · Galaxy S23: I want my keyboard to tell me the tone of a message before I send it. It has to be instant and can't slow my keyboard down; I send about a hundred messages a day.
- **S59** · OnePlus 15: When I highlight a sentence in my reference manager, I want it to tag the drugs and diseases in it within a few seconds. A couple of dozen times a day.
- **S60** · Galaxy S23: I want a shortcuts app where I type what I want and it runs the right action on my phone. I'd use it about twenty times a day and can wait a few seconds.

## How to answer

Answer in three messages, so nothing is cut off: S01-S20 now, S21-S40 when I reply "continue", S41-S60 after the next "continue". Before the first block you may write a short paragraph on your assumptions (at most 200 words); after that, output only blocks in exactly this format, one per request:

```
SCENARIO S01
PICK: C047
KEPT: C001 C002 C005
FILTERED: C003=P C004=MPE C006=D
CANNOT_RUN: C099 C100
END
```

Put every option C001-C104 in exactly one of KEPT (works), FILTERED (runs but doesn't work; after `=` list every reason, using the letters M, P, D, E) and CANNOT_RUN. Write `PICK: NONE` if nothing works. Use an empty list (e.g. `CANNOT_RUN:`) when a group is empty.
