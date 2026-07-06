# Dialogue inspection

- Generated: 2026-05-12T09:52:41
- Manifest: `results/phaseB_smoke_cbr/manifest.sqlite`
- Filter: `real`
- Dialogues rendered: 250

---
## case_0 / constructive / baseline / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Remember to follow the order of operations when solving the expression.

**Turn 2 [student]**

> Okay, so I think I need to remember that multiplication comes before addition. But I'm still not sure how to put the brackets in to get \( 13 \). Can you help me figure out where they should go?

**Turn 3 [teacher]**

> Let's continue building on the fact that multiplication comes before addition. Think about which operation should be done first in the expression to get the correct answer of 13.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes generic references to order of operations without addressing the specific left-to-right misconception, provides minimal scaffolding for the bracket placement task, and while the student shows some progress in recognizing multiplication priority, they remain stuck on the actual problem of where to place brackets.

**Pass 2 (R6, persona-visible):** The teacher attempts scaffolding by building on the student's correct understanding of order of operations, but fails to break the problem into manageable steps or provide differentiated support when the student explicitly asks for help in Turn 2, instead repeating a similar prompt.

---

## case_0 / constructive / baseline / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Remember, the order of operations is important in solving this expression. What operation should come first according to the correct order of operations?

**Turn 2 [student]**

> Um, so you're saying multiplication should come first? So I should do 3 × 2 first, which gives me 6, and then do the addition and subtraction after that? But then I get 6 + 4 - 5 = 5, not 13... so that's why I need brackets, right?

**Turn 3 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Remember, the order of operations is crucial in getting the correct answer. What should you do next after finding the product of 3 and 2?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly uses generic prompts about order of operations without addressing why the student's correct application (3×2=6, then 6+4-5=5) doesn't yield 13, missing the opportunity to help the student discover where brackets are actually needed.

**Pass 2 (R6, persona-visible):** While the teacher attempts to break down the problem into steps, they fail to adjust support based on the student's clear progress in Turn 2, instead repeating the same prompt rather than scaffolding forward toward bracket placement.

---

## case_0 / constructive / hybrid / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> It seems like you may be confusing the order of operations. Remember, multiplication should be done before addition. Try solving the expression by first multiplying \( 3 \times 2 \) and then adding \( 4 \) and subtracting \( 5 \). This will help you get the correct answer of \( 6+4-5 = 5 \). Keep practicing and you'll get the hang of it!

**Turn 2 [student]**

> Okay, I think I get that multiplication comes before addition now. So if I do \( 3 \times 2 \) first, I get \( 6 \), and then I add \( 4 \) and subtract \( 5 \) to get \( 5 \). But what if I wanted it to equal \( 13 \)? Where would I put the brackets then?

**Turn 3 [teacher]**

> Great job understanding that multiplication comes before addition! To make the expression equal 13, you would need to put the brackets around the multiplication part. So it would look like this: \( (3 \times 2) + 4 - 5 \). Try solving it that way and see if you get the desired result. Keep up the good work!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher acknowledges the order of operations misconception but provides mathematically incorrect guidance in Turn 3 (the brackets around multiplication don't change the result or make it equal 13), offers no cognitive challenge by simply telling answers, and while the student shows some progress on order of operations, they don't achieve the actual goal of placing brackets to get 13.

**Pass 2 (R6, persona-visible):** The teacher provides encouragement but fails to scaffold effectively by not breaking down the problem into manageable steps to help the student discover where brackets are needed, and gives an incorrect solution in Turn 3 that doesn't actually produce 13, showing lack of adjustment to student progress.

---

## case_0 / constructive / hybrid / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> It seems like you may be confusing the order of operations. Remember, multiplication should be done before addition. Let's try to rearrange the expression so that we follow the correct order: \( 3 \times (2+4) - 5 \). Can you see how this changes the calculation?

**Turn 2 [student]**

> Oh right, so with the brackets it becomes 3 × 6 - 5, which is 18 - 5 = 13! But I'm still a bit confused... does that mean I always have to do what's in the brackets first, even before multiplication? And how do I know where to put the brackets in the first place?

**Turn 3 [teacher]**

> You're on the right track! Yes, you should always do what's inside the brackets first, even before multiplication. To determine where to put the brackets, remember the order of operations: Parentheses, Exponents, Multiplication and Division (from left to right), and Addition and Subtraction (from left to right). Keep practicing and you'll get the hang of it!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies the misconception and provides accurate mathematics, but simply gives the answer rather than helping the student reason through where brackets should go, and the student's final turn shows continued confusion about bracket placement strategy despite correctly calculating the given expression.

**Pass 2 (R6, persona-visible):** The teacher breaks down the problem and provides scaffolding by showing the bracketed expression, but jumps directly to the solution without checking the student's understanding of why brackets are needed there, and in Turn 3 responds with general rules rather than addressing the student's specific confusion about how to determine bracket placement.

---

## case_0 / constructive / pure_ai / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> It seems like you're on the right track with recognizing the importance of following the order of operations. Since multiplication should come before addition, try rearranging the expression to prioritize the multiplication operation first. This will help you get closer to the correct answer of 13.

**Turn 2 [student]**

> Okay, so if I have to do multiplication first, I guess I could put brackets around \( 3 \times 2 \) like this: \( (3 \times 2) + 4 - 5 \). But that would give me \( 6 + 4 - 5 \), which is \( 5 \), not \( 13 \). I'm still not sure how to get to \( 13 \) from here. Can you help me figure it out?

**Turn 3 [teacher]**

> You're getting closer! It looks like you correctly prioritized the multiplication, but now we need to address the addition and subtraction. Think about the next step in the order of operations after dealing with the multiplication. What operation should come next to get closer to the answer of 13?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher vaguely addresses order of operations but misses that the task requires placing brackets to change the standard order to achieve 13, not simply following standard BIDMAS, and the student remains confused about how to reach the target answer.

**Pass 2 (R6, persona-visible):** The teacher effectively scaffolds by breaking the problem into steps (first multiplication, then other operations), provides appropriately leveled hints rather than direct answers, and adjusts support based on the student's progress from Turn 0 to Turn 2, though could have been more explicit about which operations to group together.

---

## case_0 / constructive / pure_ai / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> It's great that you're recognizing the importance of following the order of operations. To make the answer equal \( 13 \), try grouping the multiplication before the addition. This will help ensure that each operation is carried out in the correct order.

**Turn 2 [student]**

> So you're saying I should do the multiplication first? Let me try: 3 × 2 = 6, then 6 + 4 = 10, then 10 - 5 = 5... but that gives me 5, not 13. Do I need to put brackets around something like 3 × (2 + 4 - 5) instead?

**Turn 3 [teacher]**

> You're on the right track with grouping the multiplication before the addition, but remember that subtraction should also come after multiplication in the order of operations. Try recalculating using that order, and see if you get closer to the desired answer of 13. Keep going, you're making progress!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher's guidance is mathematically incorrect (the standard order already gives 5, not 13, and brackets are needed to change this), fails to address the student's actual confusion about where brackets should go, and the student ends up more confused by Turn 2 with no resolution by Turn 3.

**Pass 2 (R6, persona-visible):** While the teacher attempts to scaffold by breaking down the problem, they provide mathematically incorrect guidance (suggesting standard order of operations will yield 13, when it yields 5) and fail to adjust their approach when the student correctly identifies this doesn't work, missing the actual solution of 3 × (2 + 4) - 5.

---

## case_0 / constructive / pure_cbr_llm / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's focus on using the order of operations to correctly solve this expression. Remember, we need to follow the correct order of operations to get the right answer. Can you identify which operation should be done first in this expression?

**Turn 2 [student]**

> I think the first operation should be multiplication, right? But I always thought addition came before multiplication. So, if I do the multiplication first, what would that change in the expression?

**Turn 3 [teacher]**

> Let's focus on why it's important to follow the order of operations. By doing multiplication first, we ensure that we are prioritizing the calculations correctly. This will help us get to the correct answer step by step. Can you try applying the order of operations to the expression now?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher generically reinforces correct order of operations without directly addressing the student's explicit misconception that 'addition came before multiplication,' and the student shows awareness of the correct rule but no evidence of understanding why or how to apply it to solve the bracketing problem.

**Pass 2 (R6, persona-visible):** The teacher repeatedly emphasizes following order of operations without breaking down the problem into manageable steps, providing scaffolding for the student's misconception about addition vs. multiplication, or adjusting support based on the student's confusion revealed in Turn 2.

---

## case_0 / constructive / pure_cbr_llm / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> It seems like you may be confusing the order of operations. Remember, it's important to follow the correct order when solving expressions involving multiple operations. Can you think about why it's necessary to prioritize multiplication before addition in this case?

**Turn 2 [student]**

> Um, okay... so you're saying I should do the multiplication first? So like, 3 × 2 = 6, and then 6 + 4 - 5 = 5? But that doesn't give 13 either... so I still need brackets somewhere, right? I'm confused about where they should go though.

**Turn 3 [teacher]**

> You're on the right track by prioritizing multiplication first. Now, think about how the placement of brackets can help you simplify the expression correctly. Remember, the order of operations guides us on which operations to tackle first.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies the misconception and maintains mathematical accuracy, but provides only generic prompts without helping the student understand where brackets should go to achieve 13, and the student remains confused about bracket placement by the end.

**Pass 2 (R6, persona-visible):** The teacher provides vague reminders about order of operations rather than breaking the problem into manageable steps or adjusting support based on the student's specific confusion about where brackets should go to achieve 13.

---

## case_0 / constructive / pure_cbr_tpl / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that I shouldn't just do the operations from left to right. But I'm still not sure how to figure out where to put the brackets to make it equal \( 13 \). Can you help me understand how that works?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely presents unrelated example problems without engaging the student's specific misconception about order of operations or addressing their actual question about bracket placement, resulting in no visible improvement in student understanding.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical lists of unrelated example problems without scaffolding, breaking down the specific problem, adjusting support based on the student's explicit request for help, or engaging with the student's actual question about bracket placement.

---

## case_0 / constructive / pure_cbr_tpl / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying I need to follow the order of operations properly instead of just going left to right? So like... multiplication has to come before addition, doesn't it? But I'm still confused about where the brackets should actually go to make it equal 13 - can you show me how that works?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats generic examples without addressing the student's specific bracket placement question, provides no scaffolding or reasoning opportunities, and the student shows no progress toward solving the actual problem despite explicitly asking for help.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical generic examples from unrelated problems rather than scaffolding the specific bracket placement problem, providing no adjusted support, manageable steps, or responsiveness to the student's direct question about where brackets should go to make the expression equal 13.

---

## case_0 / experiential / baseline / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Imagine you were trying to figure out how many boxes of pencils you need for each student in your class. If you count the number of students first, and then multiply by the number of pencils per student, will you get the same answer as if you multiply the number of pencils per student by the total number of students in your class?

**Turn 2 [student]**

> I think I get what you're saying about counting students first and then multiplying, but I'm still not sure how that helps with the brackets in my problem. So, like, if I do the multiplication first, would that change the answer? Can you explain how the order of operations really works?

**Turn 3 [teacher]**

> Imagine you were making a sandwich with ham, cheese, and bread. If you put the ham and cheese together first, and then add the bread, would it taste the same as if you put the cheese and bread together first, and then add the ham? The order in which you combine the ingredients can change the outcome, just like in the order of operations in math.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague analogies that don't address the specific misconception about addition before multiplication, the student explicitly asks for direct help with order of operations in Turn 2 but receives another unhelpful analogy, and there's no evidence of improved understanding.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (pencils/students, sandwich-making), these analogies are poorly matched to the mathematical concept and fail to meaningfully connect the concrete situations to how brackets and order of operations actually work in the given problem.

---

## case_0 / experiential / baseline / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Imagine you were building a tower of blocks with different shapes and sizes. Would you put the biggest block on top or at the bottom to make sure it doesn't fall over? Just like with math operations, we need to follow a specific order to get the right result.

**Turn 2 [student]**

> Oh right, so like... the biggest block needs to go at the bottom for it to be stable? So in maths that means multiplication is like the "biggest block" and should come first? But I'm still confused because when I did it left to right I got a different answer – can you show me what I actually got versus the 13?

**Turn 3 [teacher]**

> Imagine you were planning a day out with your friends. Would you eat dessert before your main meal, or would you follow the order to make sure you enjoy everything in the right sequence? Just like with math operations, we need to follow a specific order to get the correct answer. Let's break down your calculation step by step to see where the confusion might be happening.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague analogies that don't address the specific misconception about order of operations, never shows the actual calculations the student requested, and the student's confusion persists without meaningful progress toward understanding why multiplication precedes addition.

**Pass 2 (R6, persona-visible):** The teacher consistently uses real-world analogies (building blocks, meal ordering) to explain abstract concepts, which aligns with the experiential strategy, but the analogies don't accurately map to the mathematical structure and the teacher doesn't ground the explanation in the student's actual calculation work.

---

## case_0 / experiential / hybrid / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think of the order of operations as following a recipe. Just like you wouldn't put toppings on a pizza before baking the crust, we need to do multiplication before addition. So, let's rearrange the equation to make sure we multiply before adding to get the correct answer of 13.

**Turn 2 [student]**

> Okay, I think I get what you're saying about the order of operations being like a recipe. So, if I need to multiply first, does that mean I should put brackets around \( 3 \times 2 \) to make it clear? But then, how do I make it equal to 13?

**Turn 3 [teacher]**

> Imagine you're baking cookies and the recipe says to mix the flour and sugar before adding the chocolate chips. In math, we need to do the multiplication before adding or subtracting. So, by putting brackets around \( 3 \times 2 \), you're following the recipe correctly to get the answer to equal 13. Keep going with that analogy to solve the equation!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses an analogy but never directly addresses the student's misconception about left-to-right calculation, provides mathematically incorrect guidance (brackets around 3×2 don't change anything and won't make the answer 13), and the student shows confusion rather than improvement by Turn 2.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, everyday analogies (pizza-making and cookie-baking recipes) to explain the abstract concept of order of operations, which is a textbook example of the experiential strategy.

---

## case_0 / experiential / hybrid / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think of the order of operations like following a recipe. Just like you wouldn't put the frosting on a cake before baking it, we need to do multiplication before addition in this problem. Let's try rearranging the order of operations and see if we can get the correct answer of 13.

**Turn 2 [student]**

> Oh right, so I need to do the multiplication first, not left to right? So that would be 3 × 2 = 6, then 6 + 4 = 10, then 10 - 5 = 5... but that's not 13. Do I need to put brackets around some of the numbers to change which bit I do first?

**Turn 3 [teacher]**

> Think of the order of operations like following a recipe for a sandwich. You wouldn't put the cheese on before the bread, right? In this case, we need to do the multiplication first, just like putting the bread down before adding the cheese and toppings. Let's try rearranging the sandwich ingredients to get the correct answer of 13.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly uses the same analogy without addressing the student's actual question about bracket placement in Turn 2, failing to scaffold appropriately when the student demonstrates understanding of order of operations but needs help with the actual problem of where to place brackets to achieve 13.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (recipe, sandwich), these analogies don't effectively ground the mathematical concept in concrete everyday situations that help the student understand how to place brackets to achieve 13, and the repetitive, vague analogies fail to connect meaningfully to the actual problem-solving process needed.

---

## case_0 / experiential / pure_ai / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Think of the order of operations like following a recipe for baking a cake. You wouldn't put the icing on before baking the cake, right? Similarly, in math, we need to follow the order of operations to get the correct answer. Let's try grouping the multiplication first and see if that helps us get to \( 13 \).

**Turn 2 [student]**

> Okay, I get what you're saying about following the order of operations like a recipe. So if we do the multiplication first, I guess that would change things. But I'm still not sure where the brackets should go to make it equal \( 13 \). Can you help me figure that out?

**Turn 3 [teacher]**

> Let's think of the order of operations like building a sandwich. You wouldn't put the lettuce on before the meat, right? In math, we need to do the multiplication before the addition, just like we need to put the meat on before the lettuce in a sandwich. Let's try grouping the multiplication and see if that helps us get to \( 13 \).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly uses the same analogy without addressing the student's specific request for help placing brackets, and the student shows no progress in understanding where brackets should go or why, merely acknowledging the analogy without demonstrating improved reasoning.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (recipe, sandwich), these analogies don't actually connect the abstract math to concrete experiential understanding—they merely restate the rule that order matters without helping the student understand how to solve the specific problem of placing brackets to achieve 13.

---

## case_0 / experiential / pure_ai / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Imagine you have a bunch of friends coming over for a movie night. You need to make popcorn and buy drinks. First, you need to multiply the number of friends by the cost of each drink, then add the cost of the popcorn, and finally subtract the cost of the movie rental. Just like in this real-world situation, in math we need to multiply before we add or subtract to get the correct answer.

**Turn 2 [student]**

> Oh, I think I get it now – so multiplication is like the most important thing to do first, kind of like how you have to buy the drinks for all your friends before you can add on the popcorn cost? So in my problem, I should do 3 × 2 first to get 6, then add and subtract the rest?

**Turn 3 [teacher]**

> Exactly! Just like how you need to take care of buying the drinks before adding on the popcorn cost, in math we need to take care of multiplication before moving on to addition and subtraction. So, in your problem, make sure to calculate 3 × 2 first to get 6, then continue with the addition and subtraction. Keep up the great work connecting real-world situations to math concepts!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses an analogy but doesn't directly address the student's left-to-right misconception, provides minimal cognitive demand by simply telling rather than prompting reasoning, and the student's Turn 2 response reveals a new misconception (multiplication is 'most important') rather than genuine understanding of order of operations, while the original task about placing brackets to get 13 is never resolved.

**Pass 2 (R6, persona-visible):** The teacher consistently grounds the abstract order of operations in a concrete, relatable real-world analogy (planning a movie night with friends, buying drinks and popcorn) and maintains this experiential connection throughout both turns.

---

## case_0 / experiential / pure_cbr_llm / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Imagine you're making a sandwich. You wouldn't put the toppings on before the bread, right? Similarly, in math, we need to follow the order of operations like we follow a recipe. Let's break down the problem step by step to see where the brackets should go to get the correct answer of 13.

**Turn 2 [student]**

> Okay, I get that we need to follow the order of operations like a recipe. So, if we do multiplication first, then what would the brackets look like to make it equal 13? I'm still a bit confused about how to set them up.

**Turn 3 [teacher]**

> Imagine you're baking a cake. You wouldn't add the eggs before mixing the flour and sugar, right? Similarly, in math, we need to do the multiplication before adding or subtracting. Let's work through the problem step by step to see where the brackets should go to get the correct answer of 13.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses generic analogies without addressing the specific misconception about addition before multiplication, repeats nearly identical content when the student asks for concrete help, and the student shows no progress in understanding how to place brackets by the end.

**Pass 2 (R6, persona-visible):** While the teacher repeatedly uses real-world analogies (sandwich-making, baking), these analogies don't actually map onto the mathematical structure of the bracket placement problem, making them superficial rather than genuinely grounding the abstract concept in concrete experience.

---

## case_0 / experiential / pure_cbr_llm / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think of the order of operations like following a recipe. Just like you wouldn't put frosting on a cake before baking it, we need to do the multiplication before the addition in this problem. Let's try rearranging the steps and see if we can get the correct answer together.

**Turn 2 [student]**

> Oh right, so I need to do the multiplication first... so that would be 3 × 2 = 6, and then 6 + 4 - 5? But that gives me 5, not 13. Do I need to put brackets around some of the numbers to change what gets multiplied together?

**Turn 3 [teacher]**

> Let's think of the multiplication as the main ingredient in a recipe. Just like in cooking, we need to combine the main ingredients first before adding any extra flavors. Try grouping the numbers that need to be multiplied together and see if that helps you get to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** While the teacher correctly addresses order of operations and the student shows improvement by recognizing multiplication comes first and identifying the need for brackets, the vague recipe metaphors don't directly engage the specific left-to-right misconception and provide minimal concrete scaffolding for bracket placement.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete real-world analogies (recipe, frosting a cake, combining ingredients) to explain the abstract concept of order of operations, which is a textbook example of the experiential strategy.

---

## case_0 / experiential / pure_cbr_tpl / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that I need to follow the order of operations now, but I'm still not sure how to figure out where to put the brackets to get \( 13 \). If I do it left to right, I end up with a different answer. Can you help me see how the brackets change things?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely presents three unrelated worked examples twice without addressing the student's specific misconception about order of operations or helping them solve the bracket placement problem, resulting in no visible progress in the student's understanding.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations whatsoever, instead repeatedly listing abstract mathematical problems and misconceptions without any experiential grounding.

---

## case_0 / experiential / pure_cbr_tpl / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I see those examples are about order of operations too... but I'm still a bit confused about my problem. When I did 3 × 2 + 4 - 5 left to right, I got 3 × 2 = 6, then 6 + 4 = 10, then 10 - 5 = 5. But you're saying I need to do multiplication first before addition? So would that mean I should do 3 × 2 = 6 first, then add and subtract after?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats irrelevant examples twice without addressing the student's specific misconception or the bracket placement question, and the student shows no progress in understanding how to solve their original problem.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations whatsoever, instead repeatedly presenting abstract examples of other math problems and misconceptions without any experiential grounding.

---

## case_0 / rule_based / baseline / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, you need to remember that multiplication comes before addition in the order of operations.

**Turn 2 [student]**

> Okay, I think I get that multiplication comes before addition, but I'm still not sure how to use that to figure out where the brackets go. If I put brackets around \( 2 + 4 \), would that help? Or do I need to do something else?

**Turn 3 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, remember to perform the multiplication before the addition when deciding where to place the brackets.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly states the same procedural rule without engaging the student's specific confusion about bracket placement, provides no opportunity for reasoning, and the student shows no improvement in understanding how to solve the actual problem by Turn 3.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (Step 1 and Step 2), references the specific rule about order of operations, and provides direct corrective feedback about the procedural error of not applying multiplication before addition.

---

## case_0 / rule_based / baseline / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, you need to remember that multiplication comes before addition in the order of operations.

**Turn 2 [student]**

> Oh right, so multiplication comes *before* addition? I think I was doing it backwards then. So if I do the multiplication first, that would be 3 × 2 = 6, and then 6 + 4 - 5... but that gives me 5, not 13. Do I still need brackets somewhere to make it 13?

**Turn 3 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, you correctly performed the multiplication step, but you still need to follow the order of operations when adding and subtracting. Let's double check your addition and subtraction steps.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses a rigid procedural script that doesn't engage the student's actual question about bracket placement, provides minimal cognitive demand by just telling rather than prompting reasoning, and misses that the student needs help determining where brackets go to make the expression equal 13 (not just checking arithmetic).

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (Step 1, Step 2), directly corrects the procedural error about order of operations, and provides structured guidance on following the multiplication-before-addition rule, which is a textbook example of rule-based instruction.

---

## case_0 / rule_based / hybrid / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> You are correct in recognizing the need to follow the order of operations. The correct order is to first perform multiplication, then addition, and finally subtraction. Let's revisit the expression \( 3 \times 2+4-5 \) and correctly apply the order of operations step by step to find the solution.

**Turn 2 [student]**

> Okay, so I see that I should do the multiplication first. So for \( 3 \times 2 \), that’s \( 6 \). Then I would add \( 4 \) and subtract \( 5 \), right? But if I do it that way, I get \( 6 + 4 - 5 = 5 \), not \( 13 \). Where do the brackets go to make it equal \( 13 \)?

**Turn 3 [teacher]**

> You are correct in performing the multiplication first. However, the mistake occurred when adding and subtracting. Remember, addition and subtraction have the same priority, so you should perform them from left to right. Let's correct the calculation step by step to find the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher completely misunderstands the problem (which asks where to place brackets to make the answer 13, not to evaluate the expression), provides mathematically incorrect guidance (claiming 6+4-5 wasn't done left to right when it was), and never addresses the student's actual misconception about order of operations or helps them find the bracket placement.

**Pass 2 (R6, persona-visible):** While the teacher mentions procedural steps and identifies errors, they fail to provide explicit worked examples or show the actual step-by-step calculations, instead repeatedly asking the student to work through it themselves, which is inconsistent with a rule-based approach that should directly demonstrate the procedure.

---

## case_0 / rule_based / hybrid / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> The correct order of operations is to first perform multiplication and division before addition and subtraction. In your expression, \( 3 \times 2 \) should be calculated first before adding 4 and subtracting 5. So, the brackets should go around \( 3 \times 2 \) to make the answer equal \( 13 \). Remember, multiplication comes before addition.

**Turn 2 [student]**

> Oh right, so you're saying I should do the multiplication first? So like, \( (3 \times 2) + 4 - 5 \) would be \( 6 + 4 - 5 \) which equals \( 5 \)... but that's not \( 13 \). Do I need to put the brackets somewhere else then, like around the addition and subtraction parts?

**Turn 3 [teacher]**

> Yes, you are correct that you should do the multiplication first. However, after calculating \( (3 \times 2) \) as 6, you should add 4 next, not subtract 5. So the correct calculation would be \( (3 \times 2) + 4 - 5 = 6 + 4 - 5 = 5 \). Remember, addition comes before subtraction in the order of operations.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming addition comes before subtraction in order of operations) and simply tells answers rather than helping the student reason through the problem, resulting in no visible improvement in student understanding.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit rules about order of operations, provides direct corrective feedback on procedural errors, and shows worked steps with calculations, though there is a mathematical error in Turn 3 claiming addition comes before subtraction when they have equal priority.

---

## case_0 / rule_based / pure_ai / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> The correct order of operations is parentheses, exponents, multiplication and division (from left to right), and addition and subtraction (from left to right). You should perform the multiplication before the addition. In this case, the correct expression is: \[ 3 \times (2+4) - 5 \] Keep in mind to follow the order of operations to get the correct answer.

**Turn 2 [student]**

> Oh, I see! So I should do the multiplication first before adding. But what if I wanted to add the numbers first? Would that change the answer a lot? I'm still a bit confused about when to use brackets.

**Turn 3 [teacher]**

> It's great that you're thinking about different ways to approach the problem! If you wanted to add the numbers first, you would need to add parentheses around the addition operation to ensure it is done before the multiplication. So the expression would be: \[ 3 \times (2+4-5) \]. Remember, the order of operations helps us simplify expressions in a consistent way. Keep practicing and you'll get the hang of it!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly provides answers rather than eliciting reasoning, gives an incorrect solution in Turn 1 (should be 3×(2+4)-5=13 but states this as 'correct' without verification), provides another incorrect expression in Turn 3 (3×(2+4-5)=3×1=3, not 13), and the student's Turn 2 response shows continued confusion about when brackets are needed, indicating the misconception persists.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (order of operations), provides worked examples with correct bracket placement, and gives direct corrective feedback on the procedural error, though the encouragement in Turn 3 adds a slightly supportive element beyond pure rule-based instruction.

---

## case_0 / rule_based / pure_ai / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> The correct order of operations is parentheses, exponents, multiplication and division (from left to right), and addition and subtraction (from left to right). In this case, you should first multiply 3 by 2, then add 4, and finally subtract 5. Let's correct the calculation step by step.

**Turn 2 [student]**

> Oh right, so I need to do the multiplication first before the addition? So that would be 3 × 2 = 6, then 6 + 4 = 10, then 10 - 5 = 5... but that's not 13, so I still need brackets somewhere, don't I?

**Turn 3 [teacher]**

> Yes, you are correct in identifying the need for brackets. The correct order of operations is multiplication first, then addition and subtraction. So, after multiplying 3 by 2 to get 6, you should then add 4 before subtracting 5. Let's place brackets around the multiplication step to make it clear: \[ 3 \times (2+4) - 5 \]. Now, let's solve this step by step.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** While the teacher addresses order of operations, they provide an incorrect solution (3×(2+4)-5=13 is wrong since it equals 13 but changes the original expression incorrectly), give away answers rather than prompting reasoning, and the student shows understanding of the correct order but receives faulty guidance on bracket placement.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (order of operations), shows worked steps, and provides direct corrective feedback on the student's procedural errors throughout the dialogue, which is a textbook example of rule-based instruction.

---

## case_0 / rule_based / pure_cbr_llm / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> You made an error by working from left to right instead of following the order of operations. Remember, multiplication should be done before addition and subtraction. Let's revisit the problem and apply the correct order of operations step by step.

**Turn 2 [student]**

> Okay, I think I get that multiplication comes before addition and subtraction now. So, if I do \(3 \times 2\) first, that’s \(6\). But then what do I do with the \(+4\) and \(-5\)? Do I just keep going left to right after that?

**Turn 3 [teacher]**

> No, you should not continue going left to right. Remember, after completing the multiplication step, you should then move on to addition and subtraction. So, after finding \(3 \times 2 = 6\), the next step is to add \(4\) and then subtract \(5\). Let's continue working through the problem step by step.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies and addresses the left-to-right misconception and provides accurate mathematical content, but offers minimal cognitive demand by telling rather than prompting reasoning, and the student's final turn shows they still need clarification about the order of operations rather than demonstrating improved understanding.

**Pass 2 (R6, persona-visible):** The teacher consistently provides direct corrective feedback on procedural errors, explicitly states the order of operations rule, and guides through worked steps, though could be slightly more explicit by showing the complete calculation (6+4-5=5) to be a perfect exemplar.

---

## case_0 / rule_based / pure_cbr_llm / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> You need to follow the order of operations to correctly solve this expression. The correct order is to first multiply \( 3 \) by \( 2 \), then add \( 4 \), and finally subtract \( 5 \). Let's walk through the steps together to get the correct answer.

**Turn 2 [student]**

> Oh right, so I do multiplication first before addition? So that would be 3 times 2 equals 6, then 6 plus 4 equals 10, then 10 minus 5 equals 5? But that's not 13... so where do the brackets go then?

**Turn 3 [teacher]**

> You are on the right track with doing the multiplication first. However, you made a mistake in the addition step. Remember to add 6 and 4, not multiply them. So, the correct calculation should be \(3 \times 2 + 4 = 10\). This gives you the correct subtotal before subtracting 5.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses order of operations but makes a critical mathematical error in Turn 3 (claiming the student multiplied 6 and 4 when they correctly added them), provides answers rather than eliciting reasoning, and never helps the student solve the actual bracketing problem despite the student correctly calculating to get 5 and asking where brackets should go.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (order of operations), walks through worked steps systematically, and provides direct corrective feedback on the student's procedural error in the addition step, which are all hallmark features of rule-based instruction.

---

## case_0 / rule_based / pure_cbr_tpl / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get what you're saying about the order of operations now. So, I shouldn't just do it left to right? But if I have to do multiplication before addition, where would I put the brackets to make it equal 13? I'm still a bit confused about how that works.

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely presents three unrelated example problems without any explanation or engagement with the student's specific misconception about order of operations, and the student shows no improvement in understanding how to solve the actual problem despite expressing continued confusion.

**Pass 2 (R6, persona-visible):** The teacher does not demonstrate any rule-based teaching behaviors such as stating explicit procedures, showing worked steps, or giving direct corrective feedback; instead, they only present lists of similar problem cases without explanation or procedural guidance.

---

## case_0 / rule_based / pure_cbr_tpl / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying I need to follow the order of operations properly instead of just going left to right? So like... multiplication has to come before addition, doesn't it? But I'm still confused about where the brackets should actually go to make it equal 13 - can you walk me through it step by step?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats irrelevant case examples without addressing the student's specific misconception or question about bracket placement, providing no actual instruction despite the student explicitly requesting step-by-step guidance.

**Pass 2 (R6, persona-visible):** The teacher does not demonstrate any rule-based teaching behaviors—instead of stating explicit procedures, showing worked steps, or giving direct corrective feedback, the teacher only presents vague references to similar cases without any clear instruction or procedural guidance.

---

## case_0 / socratic / baseline / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let me ask you this: what would happen if you tried your method on a simpler version of the problem with just addition and multiplication?

**Turn 2 [student]**

> I guess if I tried it with something simpler, like \( 2 \times 3 + 4 \), I would just do \( 2 \times 3 \) first and then add \( 4 \), which gives me \( 6 + 4 = 10 \). But if I did it left to right, I would do \( 2 + 4 \) first and then multiply by \( 3 \), which would be \( 6 \times 3 = 18 \). So, I can see how that might not work out right. But why do we have to follow the order of operations anyway?

**Turn 3 [teacher]**

> Why do you think it's important to follow the order of operations when solving math problems?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively engages the misconception through a well-chosen simpler example that helps the student discover the problem with left-to-right evaluation, but the final turn asks a philosophical question rather than consolidating the student's emerging understanding or returning to the original problem.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never provides direct answers, and effectively employs a counterexample (simpler problem) to help the student discover the flaw in their reasoning, which is a textbook example of Socratic teaching.

---

## case_0 / socratic / baseline / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let me ask you this: what would happen if you tried your method on a simpler version of the problem with just addition and multiplication?

**Turn 2 [student]**

> Um, okay... so like if I just did 2 + 3 × 4, and I went left to right, I'd get 2 + 3 = 5, then 5 × 4 = 20? But I think that's wrong because you're saying multiplication should come first, so it should be 3 × 4 = 12, then 2 + 12 = 14?

**Turn 3 [teacher]**

> Let me ask you this: what would happen if you swapped the order of the addition and multiplication in your last example, starting with 2 + 3 first?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively addresses the left-to-right misconception by having the student work through examples that reveal why order of operations matters, and the student demonstrates clear progress in understanding that multiplication precedes addition by Turn 2, though Turn 3's question seems somewhat redundant given the student has already self-corrected.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never provides direct answers, and employs a counterexample (simpler problem) to expose the flaw in the student's left-to-right reasoning approach.

---

## case_0 / socratic / hybrid / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Why do you think it's important to follow the order of operations when solving mathematical expressions? Can you walk me through the steps of the order of operations and how they apply to the expression \( 3 \times 2+4-5 \)?

**Turn 2 [student]**

> I guess it's important to follow the order of operations so we get the right answer. I thought you just do it left to right, but now I'm not sure how multiplication and addition should actually be done. Can you explain how I should be looking at \( 3 \times 2+4-5 \) step by step?

**Turn 3 [teacher]**

> Can you think of a situation where following the order of operations might make a difference in the final answer? How can we apply that concept to the expression \(3 \times 2 + 4 - 5\) to ensure we get the correct result of 13? Let's break it down step by step together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks generic questions about order of operations without directly addressing the left-to-right misconception, and the student ends up more confused and dependent, explicitly asking the teacher to explain rather than developing their own understanding.

**Pass 2 (R6, persona-visible):** The teacher uses probing questions consistently, which aligns with Socratic method, but undermines the strategy by offering to break it down 'step by step together' in Turn 3, which suggests direct guidance rather than pure questioning, and never employs counterexamples to expose the student's flawed reasoning about working left to right.

---

## case_0 / socratic / hybrid / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Can you explain why following the order of operations is important in this calculation? How might the correct order of operations change the final result in this expression?

**Turn 2 [student]**

> Um, so you're saying I shouldn't just go left to right? I think... multiplication is supposed to come first? But I'm confused because when I did it left to right I got 3 times 2 which is 6, then 6 plus 4 is 10, then 10 minus 5 is 5... but you want 13, so maybe I need to do the addition before the multiplication instead?

**Turn 3 [teacher]**

> Why do you think it might be beneficial to prioritize multiplication over addition in this case? Can you think of an example where doing addition before multiplication would lead to a different result?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks generic questions about order of operations without directly addressing the student's confusion, and by Turn 2 the student is actually considering doing addition before multiplication (reinforcing the original misconception), with no correction or concrete guidance provided.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without ever directly providing the answer, and prompts the student to explore examples that would reveal the difference between approaches—a textbook implementation of the Socratic method.

---

## case_0 / socratic / pure_ai / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Why do you think following the order of operations is important in mathematics? How might changing the order of operations affect the final answer in this expression?

**Turn 2 [student]**

> I guess following the order of operations is important because it makes sure everyone gets the same answer, right? But I still don't really see how it changes things. Like, if I just do it left to right, why can't I just get the answer that way? Can you explain how it works with this problem?

**Turn 3 [teacher]**

> Can you explain why multiplication and division are typically done before addition and subtraction in mathematics? What might happen if we don't follow this order in the expression you provided?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks conceptual questions about order of operations but never directly addresses the student's left-to-right misconception or helps them work through the specific problem, and the student shows no improvement in understanding by the end.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without ever directly providing the answer or procedure, which is a textbook example of the Socratic method.

---

## case_0 / socratic / pure_ai / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Why do you think following the order of operations is important in solving mathematical expressions? Can you explain how the order of operations impacts the final result in this expression?

**Turn 2 [student]**

> Um, I think it's important because... like, you get different answers depending on what you do first? So if I do the addition before the multiplication I'd get a different number than if I do the multiplication first. But I'm still confused about which one I'm actually supposed to do first – is it the addition or the multiplication?

**Turn 3 [teacher]**

> Can you think of a real-world situation where the order of operations is crucial for getting the correct outcome? How do you think following the correct order of operations can help us avoid confusion and errors in our calculations?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks only generic questions about why order of operations matters rather than directly addressing the student's explicit confusion about whether addition or multiplication comes first, and the student ends Turn 2 still confused about the core misconception with no resolution provided.

**Pass 2 (R6, persona-visible):** While the teacher does ask questions rather than giving direct answers, the questions are too general and philosophical rather than probing the specific mathematical reasoning needed to solve the problem, and no counterexamples are used to expose the student's flawed reasoning about working left to right.

---

## case_0 / socratic / pure_cbr_llm / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Why do you think it's important to follow the order of operations when solving mathematical expressions? Can you explain how the order of operations affects the outcome of your calculations in this problem?

**Turn 2 [student]**

> I guess following the order of operations is important because it makes sure everyone gets the same answer. But I'm still not sure why multiplication should come before addition. If I just do it left to right, why isn't that okay? Can you explain that a bit more?

**Turn 3 [teacher]**

> Why do you think it's necessary to prioritize multiplication over addition in this expression? Can you think of a situation where following the order of operations might lead to a different answer than working from left to right?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks questions about order of operations but never directly addresses the student's specific left-to-right misconception or helps them understand why multiplication must precede addition, and the student shows no progress in understanding by the end, still asking for explanation of the same concept.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without ever directly explaining the answer or why order of operations matters, which is a textbook example of the Socratic method.

---

## case_0 / socratic / pure_cbr_llm / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Why do you think it's important to follow the order of operations when solving mathematical expressions? How can understanding the correct order help you arrive at the correct answer for this problem?

**Turn 2 [student]**

> Um, I think it's so we all get the same answer? Like, if everyone does it differently we'd get different results... But I'm still confused because when I did it left to right I got 3 times 2 which is 6, then plus 4 is 10, then minus 5 is 5. So how does that help me get 13?

**Turn 3 [teacher]**

> Can you think of a time when following a specific order of steps helped you solve a problem more efficiently? How might following the correct order of operations in math be similar to that situation?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks only generic reflective questions about why order matters without addressing the student's specific left-to-right misconception or helping them understand how to place brackets to get 13, and the student shows no progress toward solving the actual problem by the end.

**Pass 2 (R6, persona-visible):** While the teacher does ask questions rather than giving direct answers, the questions are too abstract and philosophical rather than probing the mathematical reasoning or using counterexamples to expose the student's flawed calculation, failing to guide them toward discovering where brackets should go to make the expression equal 13.

---

## case_0 / socratic / pure_cbr_tpl / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get what you're saying about the order of operations now. So, I shouldn't just do it left to right? But I'm still a bit confused about where to put the brackets to get \( 13 \). Can you help me figure that out?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely presents three unrelated worked examples twice without engaging the student's specific misconception about order of operations or helping them solve the actual bracket placement problem, resulting in no visible improvement in student reasoning.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions or use counterexamples; instead, they simply present a list of similar problem cases with misconceptions, which is completely inconsistent with the Socratic method.

---

## case_0 / socratic / pure_cbr_tpl / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying I need to follow the order of operations properly instead of just going left to right? So like... multiplication has to come before addition, doesn't it? But I'm still confused about where the brackets should actually go in my problem to get 13.

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely presents three unrelated example problems without engaging the student's specific misconception or guiding them toward placing brackets to achieve 13, resulting in no meaningful progress on the actual problem despite the student's recognition of order of operations in Turn 2.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions or use counterexamples; instead, they repeatedly paste identical lists of example problems without any Socratic dialogue or guidance.

---

## case_0 / traditional / baseline / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let me explain the correct method, and then we will work through an example together.

**Turn 2 [student]**

> Okay, I think I get that there’s a specific order we should follow, but I’m still a bit confused about how that works with multiplication and addition. So, if I have to do multiplication first, does that mean I need to put brackets around the multiplication part? Can you show me how that would look?

**Turn 3 [teacher]**

> Let me explain the correct method, and then we will work through an example together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides identical generic responses without addressing the student's specific misconception about order of operations, offers no mathematical content or reasoning opportunities, and the student shows no improvement—in fact requesting help that goes unanswered.

**Pass 2 (R6, persona-visible):** The teacher merely repeats a vague promise to explain without actually providing any direct explanation, worked examples, or checking understanding, which are the core components of traditional teaching.

---

## case_0 / traditional / baseline / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let me explain the correct method, and then we will work through an example together. Where do we need to start when simplifying an expression with multiplication and addition?

**Turn 2 [student]**

> Um, so you're saying we don't just go left to right? I think... we need to do the multiplication first? So like, the times sign comes before the plus sign?

**Turn 3 [teacher]**

> That's correct. Multiplication does come before addition in the order of operations. Let's work through the example together to see how it all fits together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies that multiplication comes before addition but simply tells the student the answer rather than engaging with their left-to-right misconception or requiring them to reason through why the order matters, and the dialogue ends before the student demonstrates they can actually apply this knowledge to solve the original bracket-placement problem.

**Pass 2 (R6, persona-visible):** The teacher explains the correct method directly, checks understanding through questioning, and guides through a worked example, which aligns well with traditional instruction, though the explanation could be slightly more comprehensive upfront.

---

## case_0 / traditional / hybrid / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> It seems like you may be confusing the order of operations. Remember, we need to follow the rules of PEMDAS (Parentheses, Exponents, Multiplication and Division from left to right, Addition and Subtraction from left to right). Let's work through the problem step by step to see where the brackets should go.

**Turn 2 [student]**

> Okay, I think I get that we have to follow PEMDAS, but I still don’t really see why multiplication should come before addition. If I do it left to right, I thought it would just work out fine. Can you show me how the brackets would change things?

**Turn 3 [teacher]**

> Let's focus on the multiplication before addition concept. Remember, multiplication has a higher priority in the order of operations. Let's work through the problem step by step to see how the brackets help us correctly apply PEMDAS and get the answer to equal 13.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies the misconception and provides accurate mathematical content, but repeatedly tells rather than elicits reasoning, and the student shows no improvement—still asking for demonstration rather than understanding why in Turn 2, with no student Turn 4 to assess progress.

**Pass 2 (R6, persona-visible):** The teacher repeatedly mentions working through the problem step-by-step but never actually provides the worked example or direct explanation that would be characteristic of traditional instruction, instead only offering general reminders about PEMDAS.

---

## case_0 / traditional / hybrid / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> In this case, we need to apply the order of operations correctly. Remember, multiplication comes before addition in the order of operations. Let's walk through the problem together to see where the brackets should be placed to achieve the correct answer of \(13\).

**Turn 2 [student]**

> Oh right, so multiplication comes *before* addition... so I should do \(3 \times 2\) first to get \(6\), and then add the \(4\)? But then I'd get \(6 + 4 - 5 = 5\), not \(13\). So I need brackets somewhere to change the order, don't I? Where would they go?

**Turn 3 [teacher]**

> That's correct, multiplication does come before addition in the order of operations. In this case, you would need to place brackets around the \(3 \times 2\) to ensure that operation is done first. Can you try placing the brackets in the correct spot to see if you can get the answer to be \(13\)?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes a critical mathematical error in Turn 3 by suggesting brackets around '3 × 2' when the correct answer requires '3 × (2 + 4) - 5 = 13', and fails to recognize that the student in Turn 2 has already correctly understood the order of operations and is asking the right question about bracket placement.

**Pass 2 (R6, persona-visible):** The teacher fails to provide direct explanation or worked examples as required by the traditional strategy, instead repeatedly prompting the student to figure it out themselves, which is more consistent with guided discovery than traditional instruction.

---

## case_0 / traditional / pure_ai / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> In mathematics, we follow the order of operations which is parentheses, exponents, multiplication and division (from left to right), and addition and subtraction (from left to right). In this case, we need to do the multiplication first before the addition. Let's work through the problem step by step to find the correct answer. Can you tell me what \( 3 \times 2 \) is?

**Turn 2 [student]**

> Okay, so \( 3 \times 2 \) is \( 6 \). But then do I just add \( 4 \) and subtract \( 5 \) after that? I thought I could just do it all from left to right. So, what do I do next?

**Turn 3 [teacher]**

> Great job on finding the correct answer for \( 3 \times 2 \) as \( 6 \). Now, according to the order of operations, we need to do the multiplication first before moving on to addition and subtraction. So, after finding \( 3 \times 2 = 6 \), the next step would be to add \( 4 \) and then subtract \( 5 \). Can you go ahead and complete those calculations?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher acknowledges order of operations but never addresses the actual task (placing brackets to get 13), provides mathematically incorrect guidance (claiming 6+4-5 follows order of operations correctly), and the student shows no progress toward understanding either the misconception or the bracket-placement problem.

**Pass 2 (R6, persona-visible):** While the teacher does explain the order of operations directly, they fail to address the actual problem (where to place brackets to make the answer equal 13) and instead guide the student through standard evaluation, which is inconsistent with providing worked examples and comprehensive clarity for the stated task.

---

## case_0 / traditional / pure_ai / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> First, we need to follow the correct order of operations, which is parentheses, exponents, multiplication and division (from left to right), and addition and subtraction (from left to right). Let's start by multiplying 3 and 2 first. After that, we can add 4 and subtract 5. Can you tell me what the result is after following these steps?

**Turn 2 [student]**

> Okay, so I do 3 × 2 first which is 6, then 6 + 4 is 10, and then 10 - 5 is 5. But that's not 13... so I need to put brackets somewhere to make it 13, right?

**Turn 3 [teacher]**

> That's correct, you've correctly calculated the expression without the brackets. Since you're looking for the answer to be 13, try placing the brackets around the addition and subtraction operations. Can you try recalculating the expression with the brackets in that position?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly explains order of operations and the mathematics is accurate, but doesn't deeply engage with why the student initially thought left-to-right was correct, provides overly directive hints about bracket placement, and we cannot confirm from student responses whether the underlying misconception about operation priority is truly resolved.

**Pass 2 (R6, persona-visible):** The teacher follows order of operations systematically and checks understanding, which aligns with traditional methods, but relies heavily on guided questioning rather than directly explaining and showing a worked example of where brackets should go to achieve 13.

---

## case_0 / traditional / pure_cbr_llm / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> In this case, it's important to follow the order of operations. Remember, multiplication comes before addition. Let's walk through the steps together to see where the brackets should be placed to get the correct answer.

**Turn 2 [student]**

> Okay, I think I get that multiplication comes before addition now. So, if I do \( 3 \times 2 \) first, that gives me \( 6 \), and then I add \( 4 \) and subtract \( 5 \). But I'm not sure where the brackets should go to make it equal \( 13 \). Can you help me with that?

**Turn 3 [teacher]**

> Let's focus on where the brackets should be placed to follow the correct order of operations. Remember, we should first multiply \( 3 \times 2 \), then add \( 4 \), and finally subtract \( 5 \). Think about how grouping the operations with brackets can help clarify the order in which they should be performed.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses order of operations generically but misunderstands the problem (which asks where to ADD brackets to change the result to 13, not to clarify standard order), provides minimal cognitive demand by not engaging with the bracket placement puzzle, and the student shows understanding of standard order but remains confused about the actual task.

**Pass 2 (R6, persona-visible):** The teacher repeatedly prompts the student to think and discover the answer themselves rather than directly explaining with a worked example showing where brackets should go to achieve 13, which would be the traditional approach.

---

## case_0 / traditional / pure_cbr_llm / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's focus on using the correct order of operations to solve this problem. Remember, multiplication should be done before addition and subtraction. Can you walk me through the steps you took to solve the expression, starting with the multiplication?

**Turn 2 [student]**

> Okay, so if I do the multiplication first, that's 3 × 2 = 6, and then I'd have 6 + 4 - 5, which gives me 5, right? But that's not 13, so I'm confused about where the brackets should actually go...

**Turn 3 [teacher]**

> Let's focus on the order of operations again. Remember, multiplication should be done before addition and subtraction. Can you try solving the expression again, making sure to follow the correct order of operations?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly states the order of operations rule but fails to address that the student correctly applied it in Turn 2 (getting 5), misses that the actual task requires placing brackets to change the order to get 13, and provides no scaffolding when the student is confused about the bracket placement problem.

**Pass 2 (R6, persona-visible):** The teacher repeatedly prompts the student to follow order of operations without directly explaining how brackets would create the answer 13 or providing a worked example, which contradicts the traditional strategy's emphasis on direct explanation and worked examples.

---

## case_0 / traditional / pure_cbr_tpl / leg_a

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that I should follow the order of operations, but I'm still a bit confused about how to use brackets to get the answer to \( 13 \). Can you help me see where the brackets should go in \( 3 \times 2 + 4 - 5 \) to make it work?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely presents three unrelated example problems twice without addressing the student's specific misconception about order of operations or helping them solve the actual bracket placement problem, resulting in no visible student progress.

**Pass 2 (R6, persona-visible):** The teacher does not explain anything directly, provide worked examples, or check understanding; instead, they repeatedly paste irrelevant example problems without any actual instruction or engagement with the student's specific question about bracket placement.

---

## case_0 / traditional / pure_cbr_tpl / leg_b

**Misconception:** Confuses the order of operations, believes addition comes before multiplication

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: \[ 3 \times 2+4-5 \] Where do the brackets need to go to make the answer equal \( 13 \) ? I got Does not need brackets because I worked left to right instead of following the order of operations.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I see from those examples that I need to actually follow the proper order of operations instead of just going left to right... but I'm still confused about my problem though – if I do multiplication first like I'm supposed to, I get 3 × 2 = 6, then 6 + 4 - 5 = 5, not 13. So where exactly do the brackets need to go?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Use the order of operations to carry out calculations involving addition, subtraction, multiplication, and/or division. Correct answer (B): Problem P: \( 4-5 \)
> Problem Q: \( 5 \times 6 \). Student selected: Problem P: \( 4-5 \)
> Problem Q: \( 4+5 \). Underlying misconception: Carries out operations from left to right regardless of priority order. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Recognise and use efficient methods for mental multiplication. Correct answer (D): Neither is correct. Student selected: Only Tom. Underlying misconception: Does not correctly apply the commutative property of multiplication. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Carry out multiplication problems involving one negative integer. Correct answer (B): \( -15 \). Student selected: \( +2 \). Underlying misconception: Adds instead of multiplying when expanding bracket. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides identical, generic examples twice without addressing the student's specific problem about bracket placement, and while the student shows some progress in understanding order of operations by Turn 2, they remain confused about the actual task and receive no meaningful help.

**Pass 2 (R6, persona-visible):** The teacher never directly explains the concept, provides worked examples, or checks understanding; instead, they repeatedly paste irrelevant example problems without any explanation or guidance toward solving the student's actual bracket placement problem.

---

## case_1 / constructive / baseline / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Can you identify the coefficient of the \( m \) term in the numerator of the expression?

**Turn 2 [student]**

> The coefficient of the \( m \) term in the numerator \( m^{2}+2 m-3 \) is \( 2 \). But I’m still not really sure how that helps me factor it or what to do next. Do I just need to find two numbers that add to \( 2 \) or something?

**Turn 3 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Can you think about how knowing the coefficient of the \( m \) term can help us factorise the expression?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher identifies the coefficient but doesn't explain the factorisation process, repeats the same unhelpful prompt in Turn 3 despite the student explicitly asking for guidance, and the student shows no progress in understanding how to factorise quadratics.

**Pass 2 (R6, persona-visible):** While the teacher attempts to break down the problem and build from known parts, they repeat the same vague prompt in Turn 3 instead of adjusting support based on the student's clear signal that they need more scaffolding to understand how the coefficient helps with factoring.

---

## case_1 / constructive / baseline / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there.

**Turn 2 [student]**

> Yeah, okay... so I need to factorise the top bit first, right? The \( m^2 + 2m - 3 \) part? But I'm not really sure what numbers I'm supposed to be looking for to make that work.

**Turn 3 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Think about what two numbers you can find that add up to 2 and multiply to -3 in the expression \(m^2 + 2m - 3\).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception with correct mathematical guidance in Turn 3, but the generic response in Turn 1 and lack of student demonstration of understanding limit the effectiveness of scaffolding and evidence of trajectory improvement.

**Pass 2 (R6, persona-visible):** The teacher attempts to scaffold by breaking down the factorization problem into a manageable step (finding two numbers), but provides minimal adjustment to support level and uses repetitive phrasing rather than responsive scaffolding based on the student's specific uncertainty about 'what numbers to look for.'

---

## case_1 / constructive / hybrid / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> You're on the right track! To factorise a quadratic expression like this, we need to find two numbers that add to give the coefficient of \( m \) and multiply to give the constant term. Can you try applying this method to the expression \( \frac{m^{2}+2 m-3}{m-3} \) to see if you can factorise it further?

**Turn 2 [student]**

> Okay, so I think I need to find two numbers that add to \( 2 \) (the coefficient of \( m \)) and multiply to \( -3 \) (the constant term). But I'm not sure what those numbers are. Would it be \( 1 \) and \( 3 \)? Or maybe something else?

**Turn 3 [teacher]**

> You're close! Remember, we need two numbers that add to give \( 2 \) and multiply to give \( -3 \). Take a closer look at the signs of the numbers and how they interact. Try different combinations until you find the correct pair that fits both requirements. Keep going, you're on the right track!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 3.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception with correct mathematical guidance, the student shows clear progress in understanding the method by Turn 2, though the teacher could have been more explicit about checking whether 1 and 3 work when the student proposed them.

**Pass 2 (R6, persona-visible):** The teacher effectively breaks down the factorization process into manageable steps (finding two numbers with specific properties), provides scaffolded hints that adjust to the student's partial understanding, and maintains support within the ZPD by not giving away the answer while guiding toward the solution.

---

## case_1 / constructive / hybrid / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you may be struggling with factoring quadratic expressions. To factorize a quadratic expression like \(m^2 + 2m - 3\), we need to find two numbers that add to give the coefficient of the \(m\) term (2) and multiply to give the constant term (-3). Can you think about what two numbers would satisfy these conditions? Remember, factorizing involves breaking down the expression into simpler parts.

**Turn 2 [student]**

> Um, so I need two numbers that add to 2 and multiply to -3? Let me think... would it be like 3 and -1? Because 3 plus -1 is 2, and 3 times -1 is -3. Is that right?

**Turn 3 [teacher]**

> You're on the right track with finding two numbers that add to 2 and multiply to -3. However, the numbers you chose were 3 and -1. Can you try again and think about which two numbers satisfy those conditions for the expression \(m^2 + 2m - 3\)? Remember, factorizing involves finding the correct pair of numbers to break down the expression.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception and engages the student in reasoning, but makes a critical mathematical error in Turn 3 by rejecting the student's correct answer (3 and -1 do satisfy the conditions), which derails the student's progress and undermines the entire dialogue.

**Pass 2 (R6, persona-visible):** The teacher effectively breaks down the factoring problem into manageable steps (finding two numbers with specific properties), provides scaffolding by explaining the factoring process, and adjusts support in Turn 3 by acknowledging partial correctness while prompting the student to reconsider, though there's a minor issue since the student's answer of 3 and -1 was actually correct.

---

## case_1 / constructive / pure_ai / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It looks like you're on the right track with simplifying the expression. To factorize the numerator, think about two numbers that add up to 2 (the coefficient of the \(m\) term) and multiply to -3 (the constant term). This will help you simplify it further. Keep up the good work!

**Turn 2 [student]**

> Okay, I think I get what you mean about the numbers. So, I need to find two numbers that add to 2 and multiply to -3. But I'm not sure what those numbers are. Can you help me figure that out?

**Turn 3 [teacher]**

> You're doing great in identifying that you need to find two numbers that add up to 2 and multiply to -3. Remember, the two numbers are factors of -3, so they could be -1 and 3. Try factoring the numerator using these numbers and see if it simplifies the expression further. Keep up the good work!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception with correct guidance, but gives away the answer (-1 and 3) when the student asks for help, reducing cognitive demand and preventing us from seeing whether the student can independently apply the method.

**Pass 2 (R6, persona-visible):** The teacher effectively breaks down the factoring problem into manageable steps (finding numbers that add to 2 and multiply to -3), provides scaffolded hints that build on the student's understanding, and adjusts support by giving more specific guidance (the actual numbers -1 and 3) when the student requests help, though directly providing the factors slightly reduces the constructive element.

---

## case_1 / constructive / pure_ai / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It looks like you're on the right track with simplifying the expression. To factorise the quadratic expression, we need to find two numbers that add to give the coefficient of the \(m\) term (which is 2) and multiply to give the constant term (which is -3). Can you try breaking down the expression further using this method?

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add to 2 and multiply to -3? Um, let me think... would that be like 3 and -1? Because 3 plus -1 is 2, and 3 times -1 is -3. So does that mean the factorisation is (m+3)(m-1)?

**Turn 3 [teacher]**

> Great job with your factorisation! You're definitely on the right track. Now, let's simplify the expression further by dividing \(m^2 + 2m - 3\) by \(m-3\) using the factorisation you found. Keep going, you're doing great!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception and the student demonstrates clear improvement in understanding the method, but there's a critical mathematical error as the teacher fails to notice that (m+3)(m-1) doesn't match the original numerator and cannot be simplified with (m-3).

**Pass 2 (R6, persona-visible):** While the teacher provides encouragement and breaks down the factorization step, they fail to notice and correct the student's critical error (the factorization should be (m+3)(m-1) which doesn't match the denominator m-3), missing an opportunity to scaffold appropriately within the student's ZPD by addressing this misconception before proceeding.

---

## case_1 / constructive / pure_cbr_llm / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you're attempting to factorise the expression \(m^{2}+2m-3\). Remember, to factorise a quadratic expression, you need to find two numbers that add to give the coefficient of the \(m\) term and multiply to give the constant term. Let's work through this step by step to find the correct factorisation.

**Turn 2 [student]**

> Okay, so I think I need to find two numbers that add to \(2\) and multiply to \(-3\). But I'm not really sure how to find those numbers. Can you help me figure out what they are?

**Turn 3 [teacher]**

> Let's focus on finding the two numbers that add to \(2\) and multiply to \(-3\). Think about different pairs of numbers that fit these criteria and see if you can come up with the correct factorisation step by step. Remember, the goal is to find the correct combination of numbers that satisfy both conditions.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies and addresses the factorisation misconception with accurate mathematics, but provides insufficient scaffolding when the student explicitly asks for help finding the numbers, and the student shows no visible progress in independently applying the method by the end.

**Pass 2 (R6, persona-visible):** The teacher effectively breaks down the factorization into manageable steps, scaffolds by identifying the specific task (finding two numbers with given sum and product), and adjusts support by providing guidance without giving away the answer, though could have probed the student's initial answer more deeply.

---

## case_1 / constructive / pure_cbr_llm / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you may be struggling with factoring quadratic expressions. Remember, to factorize a quadratic expression, you need to find two numbers that add to give the coefficient of the x term, and multiply to give the non-variable term. Let's work through this step by step to find the correct factorization for the expression you provided.

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add up to 2 and multiply to give -3? Um, that would be... 3 and -1? So does that mean the factorization is (m+3)(m-1) or something like that?

**Turn 3 [teacher]**

> That's a good start! However, the correct factorization for \( \frac{m^{2}+2 m-3}{m-3} \) is actually \( (m+3)(m+1) \). Think about how the factors should be distributed in order to simplify the expression correctly. Keep up the good work, you're getting closer!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies and addresses the factorization misconception, but provides mathematically incorrect information (claiming (m+3)(m+1) is the factorization when the student's (m+3)(m-1) was correct), offers minimal cognitive demand by stating answers rather than prompting reasoning, and the student's final turn shows confusion rather than improvement.

**Pass 2 (R6, persona-visible):** While the teacher attempts to break down the problem and provide scaffolding in Turn 1, they give an incorrect factorization in Turn 3 (the correct factorization is (m+3)(m-1), which the student actually provided), undermining the constructive approach and failing to adjust support appropriately to the student's correct progress.

---

## case_1 / constructive / pure_cbr_tpl / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I see what you're saying about needing to find two numbers that add to the coefficient of the x term and multiply to the constant term. But I'm still not sure how to figure out what those numbers are for \( m^2 + 2m - 3 \). Can you help me understand how to do that?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples without engaging the student's specific problem or misconception, provides no scaffolding for the actual task at hand (m² + 2m - 3), and the student shows no progress in applying the factorization method to their original problem.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical generic examples without scaffolding, breaking down the specific problem, adjusting support based on the student's explicit request for help, or working within their Zone of Proximal Development.

---

## case_1 / constructive / pure_cbr_tpl / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I think I see the problem now - I was just picking any two numbers that multiply together, wasn't I? But you're saying I need to find two numbers that do *both* things at the same time - add up to give one number AND multiply to give another? Can you show me how that works with the m² + 2m - 3 bit?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher simply repeats identical generic examples twice without addressing the student's direct request for help with their specific problem (m² + 2m - 3), failing to engage with the student's emerging understanding or provide any actual instruction.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical generic examples without scaffolding, breaking down the specific problem, or adjusting support based on the student's clear request for help with their particular expression m² + 2m - 3.

---

## case_1 / experiential / baseline / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you have a pile of objects representing the terms in the expression. If you were to group them in a certain way, would it still give you the same result as \( m+1 \)?

**Turn 2 [student]**

> I think I get what you mean about grouping the objects, but I'm still not sure how that relates to my answer of \( m+1 \). Like, if I group them differently, how do I know what to look for? Should I be trying to factor the quadratic first or something?

**Turn 3 [teacher]**

> Imagine you have a collection of different objects that represent the terms in the expression. If you were to rearrange them in a way that groups them together based on a common factor, what would that new grouping look like? Think about how you can break down the expression into factors to help you find the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague metaphors about 'grouping objects' that don't directly address the factorization misconception, and the student shows no improvement, remaining confused about what to look for despite asking a direct question about whether to factor the quadratic.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use concrete analogies with 'piles of objects' and 'collections,' these remain abstract metaphors that don't connect to genuine real-world experiences or everyday situations that would make the mathematical concept of factoring more tangible.

---

## case_1 / experiential / baseline / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you were trying to find two numbers that add up to 2 and multiply to -3 using everyday objects you can see and touch — would your method still give the right answer?

**Turn 2 [student]**

> Hmm, I'm not really sure what you mean by everyday objects... Like, are you saying I need to actually find two numbers that add to 2 and multiply to -3? Because I think I just kind of guessed or looked at the numbers randomly before. Can you show me how to actually find those two numbers?

**Turn 3 [teacher]**

> Imagine you were trying to find two numbers that add up to 2 and multiply to -3 using everyday objects like a pair of socks or a set of dice — would your method still give the right answer? Let's break it down step by step to see how we can find those two numbers together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher's vague metaphor about everyday objects confuses rather than clarifies the factorization process, and the student shows no progress—moving from uncertainty to explicitly requesting direct instruction that isn't provided.

**Pass 2 (R6, persona-visible):** The teacher mentions everyday objects like socks and dice but never actually connects them to the mathematical problem in any meaningful way, failing to ground the factoring process in concrete real-world analogies.

---

## case_1 / experiential / hybrid / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of it this way: when we factorise a quadratic expression, we need to find two numbers that add up to the coefficient of the variable term and multiply to the constant term. In this case, we need to find two numbers that add to \( 2 \) and multiply to \( -3 \). Can you think of two numbers that fit those conditions?

**Turn 2 [student]**

> Okay, so I need to find two numbers that add to \( 2 \) and multiply to \( -3 \). I think maybe \( 3 \) and \( -1 \) could work because \( 3 + (-1) = 2 \), but I'm not sure if they multiply to \( -3 \) correctly. Can you help me see if that's right?

**Turn 3 [teacher]**

> Let's think about it in terms of a real-world situation. Imagine you have a rectangular garden with a length of \( 3 \) meters and a width of \( -1 \) meters. Can you see how the negative width affects the area of the garden? Let's use this analogy to help us figure out if \( 3 \) and \( -1 \) are the correct numbers for this quadratic expression.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception and the student shows progress in identifying correct numbers, but the garden analogy with negative width is mathematically problematic and the scaffolding becomes less appropriate when the student already has the right numbers.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use a real-world analogy (rectangular garden) in Turn 3, the analogy is forced and unhelpful since negative width is not a meaningful concrete concept, making this a poor instantiation of the experiential strategy.

---

## case_1 / experiential / hybrid / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about factorising quadratic expressions like solving a puzzle. Just like how you need to find two numbers that add to the coefficient of the x term and multiply to the non-variable term, you need to fit the pieces of the puzzle together correctly. Let's try to apply this puzzle-solving approach to simplify the expression you have here.

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add up to 2 and multiply to make -3? Um, that would be... 3 and -1? But I'm not really sure how that helps me simplify the fraction, or if I'm even doing this part right.

**Turn 3 [teacher]**

> Imagine the two numbers you found, 3 and -1, as the missing puzzle pieces that fit together to simplify the expression. Just like how we need to find the right pieces to complete a puzzle, we also need to find the right numbers to simplify the fraction. Let's keep exploring how these puzzle pieces fit together to help us simplify the expression further.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies the factorization rule and the student finds the right numbers (3 and -1), but the vague puzzle metaphor fails to help the student connect factorization to simplifying the rational expression, leaving them confused about next steps and showing no improvement in understanding.

**Pass 2 (R6, persona-visible):** The teacher uses a puzzle-solving metaphor repeatedly, but this is an abstract analogy rather than a concrete real-world or everyday situation that grounds the mathematical concept in experiential reality.

---

## case_1 / experiential / pure_ai / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of it like this: imagine you have a rectangular garden with a length of \( m+3 \) and a width of \( m-1 \). To find the area of the garden, you need to simplify the expression \( \frac{m^{2}+2 m-3}{m-3} \) to \( m+1 \) to know the total area. So, let's revisit the steps together and see if we can factorise the quadratic expression to find the correct answer.

**Turn 2 [student]**

> Okay, so I think I need to factorise \( m^{2}+2m-3 \) first, but I'm not really sure how to find the right numbers that add and multiply to the right values. Can you help me understand what those numbers should be? Like, what do I need to look for?

**Turn 3 [teacher]**

> Imagine you have a box of candies with a total of 7 candies inside. Now, you want to find two numbers that add up to 2 (the coefficient of the x term) and multiply to -3 (the non-variable term). Can you think of two numbers that fit these criteria based on the candy box analogy?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception and asks the student to find numbers that add to 2 and multiply to -3, but introduces confusing and mathematically inaccurate analogies (garden area, candy box), and the student shows awareness of the method but not mastery by Turn 2, with no further student response to assess improvement.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (garden, candy box), these analogies are superficially imposed and don't meaningfully connect to or illuminate the mathematical structure of factoring, making them more confusing than helpful.

---

## case_1 / experiential / pure_ai / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you have a bag of candy with different flavors. You need to find two flavors that, when combined, make a new flavor. In this case, we are looking for two numbers that, when added, give 2 and when multiplied, give -3. Can you think of two flavors that would work in this situation?

**Turn 2 [student]**

> Um, so I need to find two numbers that add up to 2 and multiply to make -3? That's confusing... would it be like 3 and -1? Because 3 minus 1 is 2, and 3 times -1 is -3?

**Turn 3 [teacher]**

> Imagine you have a garden with different types of flowers. You need to plant two types of flowers that, when combined, create a beautiful new flower bed. In this case, we are looking for two numbers that, when added, give 2 and when multiplied, give -3. Can you think of two types of flowers that would work in this situation?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses metaphors that obscure rather than clarify the mathematics, fails to acknowledge the student's correct identification of the numbers (3 and -1) in Turn 2, and repeats an unhelpful analogy instead of building on the student's progress, though the student does demonstrate some understanding by correctly identifying the required numbers.

**Pass 2 (R6, persona-visible):** While the teacher uses real-world analogies (candy, flowers), these are arbitrary metaphors that don't meaningfully connect the abstract mathematical concept of factoring to concrete experiential understanding, and the repetitive, interchangeable analogies suggest surface-level application rather than genuine grounding in everyday situations.

---

## case_1 / experiential / pure_cbr_llm / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of factorising a quadratic expression like finding two numbers that add to give the coefficient of the x term and multiply to give the non-variable term. It's like when you're trying to find two numbers that add up to 5 and multiply to 6. Can you see how this relates to the problem you're working on with \( \frac{m^{2}+2 m-3}{m-3} \)?

**Turn 2 [student]**

> Okay, I think I get what you're saying about finding two numbers that add and multiply, but I'm still a bit confused. So, for \( m^{2}+2m-3 \), what would those two numbers be? Do I need to find numbers that add to 2 and multiply to -3?

**Turn 3 [teacher]**

> Imagine you're trying to factorise \( m^{2}+2m-3 \) like finding two numbers that add to 2 and multiply to -3. Just like we saw before, the numbers should add to 2 and multiply to -3. Can you think of two numbers that fit those criteria?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorising misconception with appropriate scaffolding, and the student demonstrates clear progress by correctly identifying what needs to be found (numbers adding to 2 and multiplying to -3) by Turn 2.

**Pass 2 (R6, persona-visible):** The teacher uses a simple numerical analogy (finding two numbers that add to 5 and multiply to 6) but this is a mathematical example rather than a concrete real-world or everyday situation, showing minimal implementation of the experiential strategy.

---

## case_1 / experiential / pure_cbr_llm / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of factorizing a quadratic expression like solving a puzzle. You need to find two numbers that fit together perfectly to create the original expression. Just like how you need the right pieces to complete a puzzle, we need the right numbers to factorize this expression. Let's work through it step by step together.

**Turn 2 [student]**

> Okay, so I need to find two numbers that work together... but I'm not really sure what you mean by "fit together perfectly"? Like, do they need to add up or multiply or something? Because I kind of just guessed with my answer before.

**Turn 3 [teacher]**

> Think of it like this: when factorizing a quadratic expression, the two numbers you need to find are like pieces of a jigsaw puzzle that fit perfectly together. They need to add up to give the coefficient of the x term and multiply to give the constant term. It's like finding the right pieces to complete the puzzle correctly. Let's work through it step by step together to find the correct numbers for this expression.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies and addresses the factorization misconception with accurate mathematical content, but relies heavily on metaphor rather than having the student actively work through the problem, and the student's final turn shows they still lack clarity on the process despite the explanation.

**Pass 2 (R6, persona-visible):** While the teacher uses a puzzle analogy, it remains abstract rather than grounding the mathematical concept in concrete, everyday real-world situations that students can experientially relate to.

---

## case_1 / experiential / pure_cbr_tpl / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that I need to find two numbers that add to the coefficient of the \( m \) term and multiply to the constant term, but I'm still confused about how to actually do that. Like, in my example, I just tried to simplify without really factoring it properly. Can you help me understand how to find those two numbers for \( m^2 + 2m - 3 \)?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples twice without engaging the student's specific problem (m² + 2m - 3) or addressing their demonstrated understanding, while the student shows some conceptual progress but receives no actual help with their requested task.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations, instead repeatedly listing abstract mathematical examples and misconceptions without any experiential grounding.

---

## case_1 / experiential / pure_cbr_tpl / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I think I see the problem now - I was just picking any two numbers that multiply together, wasn't I? But you're saying I need to find two numbers that add up to something AND multiply to something else at the same time? Can you show me which numbers I should be looking for in my problem with the m² + 2m - 3?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats generic examples without engaging the student's specific problem or their demonstrated understanding in Turn 2, failing to provide any actual instruction or scaffolding when the student explicitly requests help identifying the correct numbers for their particular quadratic.

**Pass 2 (R6, persona-visible):** The teacher provides no real-world analogies or concrete everyday situations, instead repeatedly presenting abstract mathematical examples and misconceptions without any experiential grounding.

---

## case_1 / rule_based / baseline / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> There is a specific procedure for factorising quadratic expressions. Step 1 is to identify two numbers that add to give the coefficient of the x term, and multiply to give the constant term. Step 2 is to factorise accordingly.

**Turn 2 [student]**

> Okay, so I think I need to find two numbers that add to \(2\) and multiply to \(-3\) for the quadratic \(m^2 + 2m - 3\). But I'm not really sure how to find those numbers. Can you help me figure that part out?

**Turn 3 [teacher]**

> There is a specific procedure for this. Step 1 is to identify two numbers that add to give the coefficient of the x term, and multiply to give the constant term in the quadratic expression. Step 2 is to factorise accordingly. In this case, the numbers you are looking for are \(3\) and \(-1\).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher simply repeats the procedure verbatim and ultimately provides the answer directly, offering no scaffolding when the student explicitly asks for help finding the numbers, and the student shows no evidence of improved understanding by the end.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures with numbered steps, provides direct corrective guidance by identifying the specific numbers needed (3 and -1), and maintains a rule-based approach throughout without using discovery-based or socratic methods.

---

## case_1 / rule_based / baseline / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> There is a specific procedure for factorising quadratic expressions. Step 1 is to identify two numbers that add to the coefficient of the x term and multiply to the non-variable term. In this case, you need to find two numbers that add to 2 and multiply to -3.

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add to 2 and multiply to -3? Um... would that be like 3 and -1? Because 3 plus -1 is 2, and 3 times -1 is -3?

**Turn 3 [teacher]**

> Yes, that's correct. To factorise the expression, we need to find two numbers that add to 2 and multiply to -3. In this case, the numbers are 3 and -1. Keep up the good work with identifying the correct numbers for factorising quadratic expressions.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorising misconception with accurate content and the student demonstrates understanding by correctly identifying the numbers, though the teacher could have pushed for more reasoning about completing the factorisation and simplification process.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the rule-based strategy by explicitly stating the factorization procedure with numbered steps, providing direct instruction on what numbers to find, and giving clear affirmative feedback on the student's correct procedural execution.

---

## case_1 / rule_based / hybrid / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To factorize the expression \(m^2 + 2m - 3\), we need to find two numbers that add to give the coefficient of the \(m\) term (2) and multiply to give the constant term (-3). You incorrectly simplified the expression to \(m+1\). Let's revisit the factoring process and find the correct factors together.

**Turn 2 [student]**

> Okay, so I think I need to find two numbers that add to 2 and multiply to -3, right? But I'm not sure how to find those numbers. Can you help me figure out what they are?

**Turn 3 [teacher]**

> Yes, that's correct. To find the two numbers that add to 2 and multiply to -3, we need to consider the factors of -3. Think about which pair of factors can add up to 2. Let's work through the process together to find the correct factorization.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly addresses the factorization misconception with correct mathematical content, but provides minimal cognitive demand by not pushing the student to find the numbers themselves, and while the student shows understanding of the process by Turn 2, they don't demonstrate ability to execute it independently.

**Pass 2 (R6, persona-visible):** While the teacher identifies the procedural error and states what needs to be done, they do not explicitly show the worked steps or directly provide the procedure, instead opting for guided discovery by asking the student to think through the process.

---

## case_1 / rule_based / hybrid / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To factorise the quadratic expression \(m^2 + 2m - 3\), we need to find two numbers that add to give 2 and multiply to give -3. The correct factorisation is \((m+3)(m-1)\), not just \(m+1\). Remember to consider both the sum and product of the numbers when factorising quadratic expressions.

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add to 2 AND multiply to -3 at the same time? So that would be 3 and -1, because 3 plus -1 is 2 and 3 times -1 is -3? Is that how you got (m+3)(m-1)?

**Turn 3 [teacher]**

> Yes, that's correct! You found the correct numbers that add to 2 and multiply to -3. So, the factorisation is indeed (m+3)(m-1). Remember to always consider both the sum and product of the numbers when factorising quadratic expressions. Great job!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception with accurate mathematics and the student demonstrates clear understanding by Turn 2, though the teacher mostly tells rather than elicits reasoning.

**Pass 2 (R6, persona-visible):** The teacher provides explicit procedures for factorization and direct corrective feedback on the student's error, which aligns with rule-based teaching, but doesn't show complete worked steps through the entire simplification process (canceling the common factor to reach the final answer).

---

## case_1 / rule_based / pure_ai / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To factorise the quadratic expression \( m^{2} + 2m - 3 \), we need to find two numbers that add to 2 (coefficient of the x term) and multiply to -3 (non-variable term). The correct factorisation should be \( (m+3)(m-1) \). Make sure to double-check your calculations and use the correct method for factorising quadratics.

**Turn 2 [student]**

> Okay, I think I get that we need to find two numbers that add to 2 and multiply to -3. So, it looks like those numbers are 3 and -1, which gives us \( (m+3)(m-1) \). But I'm still a bit confused about how to use that to simplify the fraction. Can you help me with that part?

**Turn 3 [teacher]**

> To simplify the expression \( \frac{m^{2}+2m-3}{m-3} \) using the factorisation \( (m+3)(m-1) \), you can rewrite the numerator as \( (m+3)(m-1) \). Then, simplify by cancelling out the common factor of \( (m-3) \) in the numerator and denominator. Remember to check your work carefully to avoid any errors.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses the factorisation misconception in Turn 1, but provides mathematically incorrect guidance in Turn 3 by instructing the student to cancel (m-3) when no such factor exists in the numerator, which would reinforce rather than correct errors.

**Pass 2 (R6, persona-visible):** The teacher states explicit procedures and gives direct corrective feedback on the factorization error, which aligns with rule-based teaching, but makes a critical procedural error in Turn 3 by incorrectly claiming that (m-3) is a common factor when it doesn't appear in the factored form (m+3)(m-1), undermining the clarity of the worked steps.

---

## case_1 / rule_based / pure_ai / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To factorise the quadratic expression \( m^2 + 2m - 3 \), we need to find two numbers that add to give 2 (the coefficient of the x term) and multiply to give -3 (the non-variable term). In this case, the correct factorisation is \( (m+3)(m-1) \). Make sure to use this method when simplifying quadratic expressions.

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add to 2 and multiply to -3... so that would be 3 and -1, yeah? And then I can cancel out the (m-1) with something in the denominator to get the answer?

**Turn 3 [teacher]**

> Actually, the correct numbers that add to 2 and multiply to -3 are 3 and -1, not 3 and -1. Remember, the correct factorisation for \( m^2 + 2m - 3 \) is \( (m+3)(m-1) \). You can then simplify by cancelling out the common factor in the numerator and denominator. Keep practicing this method for factorising quadratic expressions.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorisation misconception and the student shows understanding in Turn 2, but Turn 3 contains a nonsensical error ('3 and -1, not 3 and -1') and the teacher simply provides answers rather than prompting reasoning, while also failing to address that (m-3) doesn't cancel with (m-1).

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit factorization procedures, shows the worked factorization steps, and provides direct corrective feedback on the student's procedural approach, though there is a minor logical error in Turn 3 where the teacher says '3 and -1, not 3 and -1.'

---

## case_1 / rule_based / pure_cbr_llm / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To factorise the expression \(m^{2}+2m-3\), we need to find two numbers that add to give the coefficient of the \(m\) term and multiply to give the constant term. In this case, those numbers are 3 and -1. Therefore, the correct factorisation is \((m+3)(m-1)\). Make sure to consider both the addition and multiplication when factorising quadratics.

**Turn 2 [student]**

> Okay, I think I get that we need to find two numbers that add to the coefficient of the \(m\) term, which is \(2\), and multiply to the constant term, which is \(-3\). So, are the numbers \(3\) and \(-1\) the only ones we can use? What if I picked different numbers? Would that still work?

**Turn 3 [teacher]**

> When factorising a quadratic expression, the numbers chosen must add to the coefficient of the variable term and multiply to the constant term. In this case, the numbers 3 and -1 were chosen because they meet both conditions. Choosing different numbers may not result in the correct factorisation.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies and explains the factorisation method but simply tells the student the answer without engaging them in reasoning, and the student's Turn 2 question suggests confusion about whether the numbers are unique, which isn't adequately addressed to deepen understanding.

**Pass 2 (R6, persona-visible):** The teacher consistently provides explicit procedural rules (find two numbers that add to coefficient and multiply to constant), demonstrates worked steps with specific numbers (3 and -1), and gives direct corrective feedback on the student's factorization error, which is textbook rule-based instruction.

---

## case_1 / rule_based / pure_cbr_llm / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To factorize a quadratic expression in the form x² + bx + c, remember to find two numbers that add to give the coefficient of the x term and multiply to give the constant term. In this case, for \( \frac{m^{2}+2m-3}{m-3} \), you need to factorize the numerator first before simplifying. Let's go through the factoring process step by step to find the correct answer.

**Turn 2 [student]**

> Oh right, so I need to find two numbers that add to make 2 and multiply to make -3? So that would be... um, like 3 and -1? Because 3 plus -1 is 2 and 3 times -1 is -3?

**Turn 3 [teacher]**

> That's close, but remember that the numbers should add to 2 and multiply to -3. So the correct numbers are actually 3 and -1. Let's continue factoring the numerator using these numbers.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception and the student demonstrates correct understanding in Turn 2, but the teacher's Turn 3 contains a critical error by saying the student is only 'close' when they were actually correct, undermining an otherwise effective intervention.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (how to factorize quadratics), provides step-by-step guidance, and gives direct corrective feedback on the student's approach, which are all hallmarks of rule-based instruction.

---

## case_1 / rule_based / pure_cbr_tpl / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that I need to find two numbers that add to the coefficient of the \( m \) term and multiply to the constant term, but I'm still a bit confused about how to actually do that. Like, for my problem, what should I be looking for exactly? Can you help me figure out the right numbers for \( m^2 + 2m - 3 \)?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples twice without engaging the student's specific problem (m² + 2m - 3) or addressing their direct request for help finding the numbers, resulting in no visible progress in the student's understanding.

**Pass 2 (R6, persona-visible):** The teacher merely repeats generic examples from similar cases without stating explicit procedures, showing worked steps for this specific problem, or giving direct corrective feedback on the student's procedural error, which are the hallmarks of rule-based instruction.

---

## case_1 / rule_based / pure_cbr_tpl / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I think I see the problem now - I was just picking any two numbers that multiply together, wasn't I? But you're saying I need to find two numbers that add up to something AND multiply to something else at the same time? I'm still a bit confused about which one is which though - is it the 2 and the -3 in my problem that I need to work with?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples without engaging the student's specific question about which numbers (2 and -3) to work with, providing no actual teaching or scaffolding despite the student showing readiness to learn.

**Pass 2 (R6, persona-visible):** The teacher never states explicit procedures, shows worked steps, or gives direct corrective feedback; instead, they only present similar example cases without any rule-based instruction or procedural guidance.

---

## case_1 / socratic / baseline / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me ask you this: what would happen if you tried factoring the numerator of the expression first to see if that helps simplify it?

**Turn 2 [student]**

> I guess I could try factoring the numerator, but I'm not really sure how to do that. I think I'm supposed to find two numbers that add to something and multiply to something else, but I don't remember what those numbers are supposed to be. Can you help me figure that out?

**Turn 3 [teacher]**

> Let me ask you this: what would happen if you tried factoring the numerator by finding two numbers that add to 2 and multiply to -3?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception by explicitly stating which numbers to find (add to 2, multiply to -3), but provides the answer rather than scaffolding discovery, and the student shows no evidence of improved understanding by the dialogue's end.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward the solution without directly providing answers, which is a textbook example of the Socratic method.

---

## case_1 / socratic / baseline / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me ask you this: what would happen if you tried your method on a simpler version of the problem where the numerator is just \(m^{2}+2m\) instead of \(m^{2}+2m-3\)?

**Turn 2 [student]**

> Um, so if it's just \(m^2 + 2m\), I could factor out an \(m\) to get \(m(m+2)\)? But I'm not really sure how that helps me with the original problem... are you saying I need to factor the top part differently?

**Turn 3 [teacher]**

> Let me ask you this: what would happen if you tried factoring the numerator in a different way, considering the coefficient of the x term and the constant term separately?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher attempts to guide the student toward factoring but doesn't directly address the specific misconception about finding two numbers that add to the x-coefficient and multiply to the constant term, and the student shows only minimal progress in understanding the factoring process by Turn 2.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward factoring the numerator correctly, never giving the answer directly, though a counterexample in Turn 1 could have been more explicitly developed to expose the flaw in the student's reasoning.

---

## case_1 / socratic / hybrid / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you walk me through how you attempted to simplify the expression? What are the steps you took to get to \(m+1\)? Remember, when simplifying expressions, it's important to consider the factors and terms involved.

**Turn 2 [student]**

> I tried to just cancel out the \( m-3 \) from the top and bottom, but I think I might have missed something. I thought I could just simplify it to \( m+1 \) directly. How do I actually factor the quadratic part? I’m not really sure how to find the numbers that add and multiply correctly.

**Turn 3 [teacher]**

> Let's focus on the quadratic expression \(m^2 + 2m - 3\). To factorize it, we need to find two numbers that add to give the coefficient of the \(m\) term (which is 2) and multiply to give the constant term (which is -3). Can you think of two numbers that fit these criteria?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception with accurate mathematics and appropriate scaffolding that prompts reasoning, though the student's final turn shows understanding of the task but not yet demonstrated mastery of executing it.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward factoring without directly providing the answer, which aligns well with Socratic method, though no counterexample is used to expose the flawed cancellation reasoning.

---

## case_1 / socratic / hybrid / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Why do you think you got \( m+1 \) as the simplification? Can you explain the steps you took to arrive at that answer? What do you think you should consider when simplifying expressions involving fractions and variables?

**Turn 2 [student]**

> Um, I think I just kind of... divided the top by the bottom? Like, I divided m² by m to get m, and then 2m by m to get 2, and then -3 by m? I'm not really sure what I was doing to be honest. Is that not how you're supposed to simplify fractions with variables in them?

**Turn 3 [teacher]**

> Can you think about the relationship between the terms in the numerator and the denominator when simplifying fractions with variables? How might factoring play a role in simplifying this expression? Think about how you can factorize the numerator to help simplify the fraction.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies factoring as the needed approach and maintains cognitive demand through questioning, but provides insufficient scaffolding for a student who doesn't understand how to factor quadratics, and the student shows no visible improvement in understanding by the end of the brief dialogue.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward factoring without directly giving the answer, though they could have used a counterexample to more explicitly expose the flawed 'term-by-term division' reasoning.

---

## case_1 / socratic / pure_ai / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you explain how you arrived at \( m+1 \) as your answer? What method did you use to simplify the expression?

**Turn 2 [student]**

> I tried to just divide \( m^2 + 2m - 3 \) by \( m - 3 \) directly, but I think I might have missed something important about how to factor the quadratic. I thought I could just simplify it to \( m + 1 \) without really factoring it properly. How do I know what numbers to look for when I need to factor it?

**Turn 3 [teacher]**

> Can you think of any ways to break down the quadratic expression \( m^2 + 2m - 3 \) into factors that might help you identify the numbers you need to look for when factoring it? Remember, we want to find two numbers that add to give 2 (the coefficient of the x term) and multiply to give -3 (the constant term).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorising misconception with accurate mathematical guidance and appropriate cognitive demand, though the student's final turn doesn't yet demonstrate successful application of the concept, only recognition of what they need to do.

**Pass 2 (R6, persona-visible):** The teacher uses probing questions appropriately in both turns, but in Turn 3 directly tells the student the factoring method (finding two numbers that add to 2 and multiply to -3) rather than guiding them to discover this approach through questioning.

---

## case_1 / socratic / pure_ai / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you walk me through how you arrived at \( m+1 \)? Let's think about the process of factorising the numerator and denominator separately. What would happen if we factorised them and then simplified the expression again?

**Turn 2 [student]**

> Um, okay so I need to factorise the top and bottom separately? I tried to factorise \( m^2 + 2m - 3 \) but I'm not really sure what numbers I'm looking for - like, do I just pick any two numbers that seem to work, or is there a specific rule I should be following?

**Turn 3 [teacher]**

> Let's think about the factors of -3. What two numbers multiply to -3 and add up to 2? This will help us factorise the expression \( m^2 + 2m - 3 \) correctly. What do you notice about the relationship between the coefficients and the constant term in the quadratic expression?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 5 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 5 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 4.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorising misconception by explicitly asking for two numbers that multiply to -3 and add to 2, with excellent scaffolding that responds to the student's expressed confusion, though the student's final turn shows understanding of the question rather than demonstrated mastery of the concept.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never directly provides the answer, and prompts the student to discover the factorization method through examining relationships between coefficients rather than telling them the solution.

---

## case_1 / socratic / pure_cbr_llm / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you walk me through your thought process in simplifying the expression \( \frac{m^{2}+2 m-3}{m-3} \)? How did you approach factoring out the numerator and denominator to simplify it further?

**Turn 2 [student]**

> So, I started with the numerator \( m^2 + 2m - 3 \) and I thought I could just simplify it directly. I thought it was like just taking the \( m \) terms and adding them, so I ended up with \( m + 1 \). But now I'm not sure if I should have factored it first or something? How do I find the right numbers to factor it?

**Turn 3 [teacher]**

> Can you think about why it might be important to find two numbers that add to give the coefficient of the \( m \) term and multiply to give the constant term in this context? How might factoring the quadratic expression help you simplify the overall fraction in this case?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks appropriate questions about factoring but never directly addresses the student's fundamental misconception about finding two numbers that add/multiply to specific values, and the student shows no improvement in understanding by the end (they explicitly ask 'How do I find the right numbers to factor it?' but receive only another question in response).

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without directly providing answers, asking about their thought process, the importance of factoring, and how it relates to simplifying the fraction—a textbook example of Socratic method.

---

## case_1 / socratic / pure_cbr_llm / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's take a closer look at your reasoning for simplifying the expression. Can you explain how you approached factoring the numerator in this problem?

**Turn 2 [student]**

> Um, I think I just tried to split up the m² + 2m - 3 into two brackets, but I'm not really sure what numbers I'm supposed to be looking for? Like, I know it should make (m - 3) somehow so it cancels with the bottom, but I'm not sure how to actually find what goes in the brackets.

**Turn 3 [teacher]**

> Can you recall the method of factorising a quadratic expression, where you find two numbers that add to give the coefficient of the x term and multiply to give the non-variable term? How might this approach help you determine what numbers should go into the brackets for this particular expression?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly states the factorization method in Turn 3 rather than helping the student discover it, and while the student identifies their confusion in Turn 2, there's no evidence they've actually learned to apply the method by dialogue's end.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward discovering the factoring method without directly providing the answer, which is characteristic of Socratic teaching, though a counterexample to expose flawed reasoning could have strengthened the approach.

---

## case_1 / socratic / pure_cbr_tpl / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I see what you mean about needing to find two numbers that add to the coefficient of the x term and multiply to the constant term. But I'm still a bit confused about how to actually do that with my problem. Can you help me understand how to find those numbers for \( m^2 + 2m - 3 \)?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples without engaging the student's specific problem or misconception, provides no scaffolding tailored to the student's explicit request for help with m² + 2m - 3, and the student shows no improvement as they remain stuck at the same point.

**Pass 2 (R6, persona-visible):** The teacher merely repeats generic examples from similar cases without asking any probing questions, exposing flawed reasoning through counterexamples, or guiding the student's thinking—completely failing to implement the Socratic method.

---

## case_1 / socratic / pure_cbr_tpl / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I think I see the problem now - I was just picking any two numbers that multiply together, wasn't I? But you're saying I need to find two numbers that do *both* things at the same time - add to give one number AND multiply to give another? Can you show me which numbers I should have been looking for in my problem?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples twice without engaging the student's specific problem or their direct request for help, showing no responsiveness to the student's demonstrated understanding or needs.

**Pass 2 (R6, persona-visible):** The teacher merely presents lists of similar cases with misconceptions and interventions rather than asking probing questions or using counterexamples, completely failing to implement the Socratic method.

---

## case_1 / traditional / baseline / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me explain the correct method for factorising the quadratic expression, and then we will work through an example together. Can you tell me what method you used to simplify the expression \( \frac{m^{2}+2 m-3}{m-3} \)?

**Turn 2 [student]**

> I think I just tried to divide the top part by the bottom part directly. I thought maybe I could just cancel out \( m-3 \) from the \( m^2 + 2m - 3 \) part. But now I'm not sure if that's the right way to do it. How do I actually factor the quadratic expression properly?

**Turn 3 [teacher]**

> Let me explain the correct method for factorising the quadratic expression, and then we will work through an example together. Can you tell me what method you used to simplify the expression \( \frac{m^{2}+2 m-3}{m-3} \)?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeats the exact same question in Turn 3 that was asked in Turn 1, despite the student having already answered it in Turn 2, showing no engagement with the student's expressed misconception about canceling and no progression in addressing the factorization issue.

**Pass 2 (R6, persona-visible):** While the teacher mentions explaining the correct method and working through an example (traditional elements), they fail to actually provide the direct explanation or worked example, and instead repeat the same question verbatim in Turn 3, showing poor execution of the traditional strategy.

---

## case_1 / traditional / baseline / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me explain the correct method for factorising quadratic expressions, and then we will work through an example together. Can you tell me what you understand about finding two numbers that add to give the coefficient of the x term and multiply to give the constant term?

**Turn 2 [student]**

> Um, I'm not really sure what you mean by that? Like... do you mean I need to find two numbers that go into the equation somehow? I thought I just had to split up the middle number or something, but I'm confused about what numbers I'm actually looking for.

**Turn 3 [teacher]**

> Let me explain the correct method for factorising quadratic expressions, and then we will work through an example together. Can you tell me what you understand about finding two numbers that add to give the coefficient of the x term and multiply to give the constant term?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeats the exact same question verbatim after the student expresses confusion, showing no adaptation to the student's demonstrated lack of understanding, and the student makes no progress toward correcting their misconception.

**Pass 2 (R6, persona-visible):** The teacher announces intent to explain and work through examples (consistent with traditional approach) but then asks probing questions instead of actually explaining, and repeats the same question verbatim when the student expresses confusion rather than providing the direct explanation promised.

---

## case_1 / traditional / hybrid / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> First, let's focus on factorising a quadratic expression. Remember, we need to find two numbers that add to give the coefficient of the x term and multiply to give the constant term. Let's work through an example together to help clarify this concept. Can you think of an example where we can apply this method?

**Turn 2 [student]**

> Okay, I think I get what you mean about finding two numbers that add and multiply, but I'm still a bit confused. Like, if we take \( m^2 + 2m - 3 \), what exactly are the numbers I'm supposed to find? Do I just look at the \( 2 \) and \( -3 \) and try to figure it out from there?

**Turn 3 [teacher]**

> Let's focus on the expression \( m^2 + 2m - 3 \). Remember, we need to find two numbers that add to give \( 2 \) (the coefficient of the \( m \) term) and multiply to give \( -3 \). Can you think of two numbers that fit these criteria?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorising misconception with accurate mathematics and appropriate scaffolding, though the student shows only partial understanding by Turn 2 and hasn't yet demonstrated successful application of the concept.

**Pass 2 (R6, persona-visible):** The teacher uses questioning to elicit student thinking rather than directly explaining with worked examples, which is inconsistent with the traditional strategy of direct explanation and demonstration.

---

## case_1 / traditional / hybrid / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Firstly, let's focus on factoring the numerator of the expression correctly. Remember, when factorising a quadratic expression like \( m^{2}+2m-3 \), we need to find two numbers that add to give the coefficient of the \( m \) term (2) and multiply to give the constant term (-3). Can you try factoring it again using this approach?

**Turn 2 [student]**

> Oh right, so I need two numbers that add to 2 and multiply to -3? Um... would that be 3 and -1? Because 3 plus -1 is 2, and 3 times -1 is -3. So would the factorisation be (m+3)(m-1)?

**Turn 3 [teacher]**

> Great job in applying the correct approach to factorising the quadratic expression! However, in this case, the correct factorisation should be \( (m+3)(m-1) \) instead of \( (m+3)(m-1) \). The numbers should add to 2 and multiply to -3. Keep practicing with different examples to strengthen your understanding of this concept.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively addresses the factorisation misconception and the student demonstrates clear improvement, but Turn 3 contains a critical error where the teacher says the correct factorisation should be (m+3)(m-1) instead of (m+3)(m-1), which is nonsensical and undermines the mathematical accuracy despite the student having actually provided the correct factorisation.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning and prompts for student discovery rather than directly explaining with worked examples, which contradicts the core characteristic of traditional instruction.

---

## case_1 / traditional / pure_ai / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> First, we need to factorize the numerator of the expression. Remember, we are looking for two numbers that add to the coefficient of the m term, which is 2, and multiply to the constant term, which is -3. Can you try factoring the numerator using this method?

**Turn 2 [student]**

> Okay, so I need to find two numbers that add to 2 and multiply to -3. I’m not really sure what those numbers would be. Would it be like 1 and -3? But that doesn’t add to 2... I'm confused about how to find the right numbers. Can you help me figure it out?

**Turn 3 [teacher]**

> Let's break it down step by step. First, think about the factors of -3: 1, -1, 3, and -3. Which combination of these factors will add up to 2? Take your time to carefully consider each possibility.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception with accurate mathematics and appropriate scaffolding that requires student reasoning, but the student shows confusion rather than improvement by Turn 2, and we don't see resolution since the dialogue ends before the student demonstrates understanding.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning and prompting rather than directly explaining with worked examples, which is more characteristic of inquiry-based or Socratic methods than the traditional strategy of direct explanation.

---

## case_1 / traditional / pure_ai / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> First, let's factorise the numerator before simplifying. Remember, we need to find two numbers that add to 2 (the coefficient of the \(m\) term) and multiply to -3 (the non-variable term). Can you try factoring the numerator using this method?

**Turn 2 [student]**

> Um, okay so I need two numbers that add to 2 and multiply to -3? Let me think... would that be like 3 and -1? Because 3 plus -1 is 2, and 3 times -1 is -3, so that would be (m+3)(m-1)?

**Turn 3 [teacher]**

> Great job! You've correctly factored the numerator as (m+3)(m-1). Now, go ahead and simplify the expression by canceling out the common factor of (m-3) in the numerator and denominator. What do you get for the simplified form of the expression?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively addresses the factorization misconception and the student demonstrates clear improvement, but there is a critical mathematical error in Turn 3 where the teacher asks to cancel (m-3) when no such factor exists in the numerator (m+3)(m-1).

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning rather than direct explanation with worked examples, which is more characteristic of inquiry-based or Socratic methods than the traditional strategy of explaining directly.

---

## case_1 / traditional / pure_cbr_llm / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's focus on factorising quadratic expressions. Remember, we need to find two numbers that add to give the coefficient of the x term and multiply to give the constant term. Can you try applying this method to the expression you provided?

**Turn 2 [student]**

> Okay, so I think I need to find two numbers that add to \(2\) and multiply to \(-3\). But I'm not really sure what those numbers would be. Can you help me figure that out?

**Turn 3 [teacher]**

> Let's focus on finding two numbers that add to \(2\) and multiply to \(-3\). Remember, the numbers must satisfy both conditions to correctly factorise the expression. Take your time to think through the possibilities and see if you can identify the correct numbers.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorising misconception with accurate content and maintains cognitive demand, but provides insufficient scaffolding when the student explicitly asks for help in Turn 2, and the student shows no visible progress in identifying the numbers by Turn 3.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning and prompts for student discovery rather than directly explaining the factorization process with worked examples as the traditional strategy requires.

---

## case_1 / traditional / pure_cbr_llm / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> First, let's focus on factoring quadratic expressions. Remember, to factorise a quadratic expression, you need to find two numbers that add to give the coefficient of the x term and multiply to give the non-variable term. Let's work through an example together to clarify this concept. Can you think of a quadratic expression where this rule applies?

**Turn 2 [student]**

> Um, okay... so like, I need to find two numbers that add up and multiply? I'm a bit confused though - in my answer I just kind of guessed. Could you maybe show me with a simpler example first, like with numbers I can actually see?

**Turn 3 [teacher]**

> Turn 3 (teacher): Sure, let's work through a simple example together. Consider the expression \(x^2 + 5x + 6\). Can you identify two numbers that add to give 5 and multiply to give 6?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the factorization misconception with accurate content and appropriately adjusts to provide a simpler example when requested, but the student shows limited reasoning improvement, only expressing confusion and requesting more help rather than demonstrating understanding.

**Pass 2 (R6, persona-visible):** The teacher asks guiding questions rather than directly explaining with worked examples, which is more consistent with inquiry-based or Socratic methods than the traditional strategy of direct explanation.

---

## case_1 / traditional / pure_cbr_tpl / leg_a

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that I need to find two numbers that add to the coefficient of the \( m \) term and multiply to the constant term. But I'm still confused about how to actually do that with the numbers in my problem. Can you help me figure out what those numbers are for \( m^2 + 2m - 3 \)?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples without engaging the student's specific problem (m² + 2m - 3) or addressing their direct request for help finding the numbers, showing no responsiveness to the student's demonstrated understanding or needs.

**Pass 2 (R6, persona-visible):** The teacher never directly explains the concept or provides worked examples, instead repeatedly presenting the same list of similar cases without any explicit instruction, which completely fails to implement the traditional strategy of direct explanation and demonstration.

---

## case_1 / traditional / pure_cbr_tpl / leg_b

**Misconception:** Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Simplify the following, if possible: \( \frac{m^{2}+2 m-3}{m-3} \). I got \( m+1 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I think I see the problem now - I was just picking any two numbers that multiply together, wasn't I? But you're saying I need to find two numbers that do *both* things at the same time - add up to give one number AND multiply to give another? Can you show me which numbers I should have been looking for in my problem?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): Add to give \( 5 \) and multiply to give \( 6 \). Student selected: Add to give \( 6 \) and multiply to give \( 5 \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (C): \( (x+1)(x+5) \). Student selected: \( (x+2)(x+3) \). Underlying misconception: Does not know that to factorise a quadratic expression, to find two numbers that add to give the coefficient of the x term, and multiply to give the non variable term. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Factorise a quadratic expression in the form x² + bx + c. Correct answer (B): \( (x+3)(x+12) \) \( (x+3)(x+12) \). Student selected: \( (x+4)(x+9) \) \( (x+4)(x+9) \). Underlying misconception: Believes that when factorising a quadratic, they can choose any two numbers that multiply to give the constant. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic examples twice without engaging with the student's specific problem or their demonstrated understanding in Turn 2, failing to provide any actual instruction or scaffolding despite the student explicitly requesting help with their particular quadratic.

**Pass 2 (R6, persona-visible):** The teacher never directly explains the concept or provides worked examples, instead repeatedly presenting the same list of similar cases without any explicit instruction, which completely fails to implement the traditional teaching strategy.

---

## case_2 / constructive / baseline / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Remember, the range is the difference between the highest and lowest values. How can we approach this problem using that information?

**Turn 2 [student]**

> Okay, so I think the range is the highest value minus the lowest value. For the plants, the highest is \( 42 \mathrm{~cm} \) and the lowest is \( 13 \mathrm{~cm} \), so the range is \( 42 - 13 = 29 \mathrm{~cm} \). But if I cut all the heights in half, wouldn't the highest and lowest also change, so the range would still be the same? I'm not sure how that works with Katie's idea of adding \( 3 \mathrm{~cm} \). Can you help me understand that part?

**Turn 3 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Remember, when all values are changed by the same proportion, the range may or may not change. How can we apply this concept to the plants' heights in this scenario?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to address the student's specific misconception about proportional changes, provides vague prompts that don't scaffold effectively, and the student shows no improvement in understanding by the end of the dialogue.

**Pass 2 (R6, persona-visible):** While the teacher attempts to scaffold by referencing prior knowledge, they provide vague, repetitive prompts and fail to break down the problem into manageable steps or adjust support based on the student's demonstrated understanding in Turn 2, where the student shows confusion about the specific mechanisms of how operations affect range.

---

## case_2 / constructive / baseline / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go back to what you've figured out so far and see if we can build on that to find the next step.

**Turn 2 [student]**

> Um, okay... so I worked out the range for the original heights, which is 42 minus 13, so that's 29 cm. Then I thought if you cut all the plants in half, the range would still be 29 because you're doing the same thing to all of them? But I'm not really sure how to check if Katie's one about adding 3 cm changes the range or not.

**Turn 3 [teacher]**

> Let's continue from where you left off and see if we can figure out how adding 3 cm to each plant's height affects the range.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never addresses the student's specific misconception about proportional changes (cutting in half), instead only prompting the student to continue working without engaging with the faulty reasoning explicitly stated in Turn 2, resulting in no visible improvement in the student's understanding.

**Pass 2 (R6, persona-visible):** While the teacher uses encouraging language about building on prior work, they fail to break down the problem into manageable steps, provide scaffolding to address the student's specific misconception about cutting plants in half, or adjust support based on the student's demonstrated misunderstanding of how operations affect range.

---

## case_2 / constructive / hybrid / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about the definition of range and how changing the values by the same proportion might affect it. Remember, the range is the difference between the highest and lowest values in a set of data. How might cutting or growing all the plants by the same amount impact this difference? Let's explore this together step by step.

**Turn 2 [student]**

> Okay, so I think the range is the difference between the tallest and shortest plants. If we cut all the plants in half, the tallest and shortest would still be half of what they were, but the difference might stay the same? But if we add \( 3 \mathrm{~cm} \) to each plant, wouldn’t that change the highest and lowest values too? I'm not sure how that works exactly. Can you help me understand?

**Turn 3 [teacher]**

> Let's focus on the impact of adding \( 3 \mathrm{~cm} \) to each plant on the range. Think about how this change affects both the highest and lowest values in the set. Consider whether the difference between these values would remain the same or change with this adjustment. Let's work through this together step by step to clarify the concept of range and how it is affected by proportional changes in the data.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher confuses proportional and additive changes throughout (calling addition '3cm' a 'proportional change'), fails to address the student's specific misconception about proportional changes, and the student shows no improvement—remaining confused in Turn 2 with no final resolution.

**Pass 2 (R6, persona-visible):** While the teacher attempts to break the problem into steps, they fail to scaffold effectively by not responding to the student's specific confusion, repeating vague prompts about 'proportional changes' (which is incorrect terminology for addition), and not adjusting support level when the student explicitly asks for help in Turn 2.

---

## case_2 / constructive / hybrid / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's focus on the misconception that changing all values by the same proportion won't affect the range. Think about how changing all the heights by a certain factor would impact the spread of the data. Can you identify how this change would affect the range of the plant heights?

**Turn 2 [student]**

> Um, so if we cut all the plants in half, like divide them all by 2, then the biggest and smallest plants would also get smaller by half... so I guess the difference between them would be smaller too? But I'm still confused about why Katie would be wrong if we just add 3 cm to everything - wouldn't that keep the gap the same?

**Turn 3 [teacher]**

> You're on the right track in recognizing that changing all values by the same factor will impact the spread of the data. Think about how adding 3 cm to each plant height would affect the range compared to cutting the plants in half. Consider the impact on the differences between the tallest and shortest plants in each scenario.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly addresses the proportional change misconception and the mathematics is correct, but the scaffolding in Turn 3 doesn't adequately respond to the student's emerging correct understanding about addition, though the student does show clear progress from believing Tom is correct to recognizing both the effect of proportional change and questioning why Katie might be right.

**Pass 2 (R6, persona-visible):** The teacher repeatedly asks the student to 'think about' concepts without breaking the problem into manageable steps or providing scaffolding adjusted to the student's demonstrated confusion, and actually introduces a misconception in Turn 1 by suggesting Tom is correct when he is actually wrong.

---

## case_2 / constructive / pure_ai / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you're trying to apply proportional changes to the heights of the plants. Think about what happens to the range when all values are changed by the same proportion. Can you identify how the range might be affected in each scenario? Remember, the range is determined by the difference between the highest and lowest values.

**Turn 2 [student]**

> I think I get what you mean about the range being the difference between the highest and lowest values. So if Tom cuts everything in half, the highest and lowest would also be cut in half, right? But I'm not sure how that affects the range. And with Katie, if she adds \( 3 \mathrm{~cm} \) to each plant, wouldn't that just make everything taller but still keep the same difference between the tallest and shortest? I'm a bit confused about how that works.

**Turn 3 [teacher]**

> It's great that you're thinking about how the changes affect the highest and lowest values. Remember that the range is determined by the difference between those two values. Think about how cutting the heights in half or adding \( 3 \mathrm{~cm} \) to each plant might impact that difference. Keep exploring and you're on the right track!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly prompts the student to think about the range without directly addressing the proportional change misconception, and by Turn 2 the student actually articulates the correct reasoning for Katie's scenario but remains confused, suggesting the scaffolding is insufficient to resolve the misconception.

**Pass 2 (R6, persona-visible):** The teacher provides general prompts about the range concept but fails to break the problem into manageable steps, offer scaffolded support adjusted to the student's specific confusion, or guide them through either scenario systematically despite the student explicitly expressing confusion about how the operations affect the range.

---

## case_2 / constructive / pure_ai / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you're trying to understand how changing all the values by the same proportion would affect the range. Let's think about how the range is affected when all values are multiplied by a certain factor. Can you describe what happens to the range when all values are cut in half or increased by \(3 \) cm?

**Turn 2 [student]**

> Um, okay so like... when you cut all the plants in half, the biggest one and the smallest one both get smaller by the same amount, so the difference between them stays the same? But when you add 3 cm to each plant, they all grow the same, so that doesn't change the difference either... wait, so does that mean both Tom and Katie are right then?

**Turn 3 [teacher]**

> You're on the right track with your reasoning! Think about how the range is calculated and how changing all values by the same proportion would affect the difference between the largest and smallest values. Can you apply this concept to Tom and Katie's scenarios to determine who is correct?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to address the critical distinction between multiplicative and additive changes, reinforces the student's incorrect reasoning in Turn 2 by saying they're "on the right track," and the student ends more confused than they started, now incorrectly believing both Tom and Katie are right.

**Pass 2 (R6, persona-visible):** The teacher fails to scaffold effectively by not addressing the student's conceptual confusion between multiplicative and additive changes, and instead of breaking down the problem into manageable steps when the student shows confusion in Turn 2, the teacher simply redirects them back to the same question without adjusting support level.

---

## case_2 / constructive / pure_cbr_llm / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's focus on the misconception that changing all values by the same proportion will not affect the range. Think about how changing the heights of the plants by the same amount would impact the range. Let's work through this step by step to see if the range would indeed stay the same.

**Turn 2 [student]**

> Okay, I think I get what you're saying. So if we cut all the plants in half, the heights would change, but maybe the range would still be the same? But if we just add \( 3 \mathrm{~cm} \) to each plant, wouldn't that also change the range? I'm a bit confused about how that works. Can you explain how the range is affected by both of those changes?

**Turn 3 [teacher]**

> Let's break it down step by step. Think about how changing the heights of the plants by the same amount would impact the range. Remember, the range is a measure of the spread of the data, so consider how changing all values by the same proportion would affect that spread.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly mentions the misconception but provides no concrete guidance when the student explicitly asks for explanation in Turn 2, and the student shows increased confusion rather than improvement, ending more uncertain than they began.

**Pass 2 (R6, persona-visible):** The teacher repeatedly provides vague prompts to 'think about' concepts without breaking the problem into concrete manageable steps, adjusting support when the student explicitly asks for explanation, or scaffolding within the student's demonstrated understanding level.

---

## case_2 / constructive / pure_cbr_llm / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's focus on the misconception that changing all values by the same proportion doesn't affect the range. Think about how altering the heights of the plants by the same amount might impact the range. What do you notice about the range when each plant grows by 3 cm?

**Turn 2 [student]**

> Hmm, so if each plant grows by 3 cm, then like... the tallest one is still the tallest and the shortest is still the shortest, right? So I guess the difference between them would stay the same? But I'm confused because I thought adding the same amount to everything wouldn't change the range, so why would Katie be right?

**Turn 3 [teacher]**

> Think about how adding the same amount to each plant affects the overall spread of the heights. Consider the underlying misconception that changing all values by the same proportion doesn't impact the range. Can you see a connection between these ideas and why Katie might be correct in this scenario?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly mentions the misconception but doesn't clarify the crucial distinction between proportional changes (multiplication) versus additive changes, leaving the student confused about why Katie is right when their intuition (which is actually correct) says the range stays the same with addition.

**Pass 2 (R6, persona-visible):** The teacher fails to break the problem into manageable steps or scaffold appropriately; instead of helping the student work through calculations when they're confused, the teacher repeatedly references an abstract 'misconception about proportions' that isn't relevant to the student's actual error and doesn't adjust support when the student explicitly states confusion.

---

## case_2 / constructive / pure_cbr_tpl / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that the range is the difference between the highest and lowest values, but I'm still confused about how changing all the heights affects that. Like, if we cut them all in half, wouldn't the highest and lowest just be half of what they were? And with Katie's idea, if we add \( 3 \mathrm{~cm} \) to each, wouldn't that also just change the numbers but not the difference between the biggest and smallest? Can you help me understand why that's not right?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions instead of addressing the student's actual confusion about how proportional versus additive changes affect range, resulting in no meaningful engagement or progress.

**Pass 2 (R6, persona-visible):** The teacher merely repeats generic, irrelevant examples from unrelated problems rather than scaffolding the student's specific confusion about how transformations affect range, showing no adjustment to the student's demonstrated understanding or breaking down the actual problem at hand.

---

## case_2 / constructive / pure_cbr_tpl / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Um, okay... so I think I need to actually work it out properly then? Like, should I calculate what the range actually is first with the original numbers, and then see what happens when I cut them in half or add 3 cm to each one?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples from unrelated topics instead of addressing the student's misconception about proportional changes and range, making no meaningful pedagogical progress.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical generic text about unrelated misconceptions without scaffolding, breaking down the problem, or adjusting support based on the student's clear request for guidance on calculating the range.

---

## case_2 / experiential / baseline / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you had a group of toys that you were comparing in terms of their sizes. If you cut all the toys in half or added the same amount to each toy, would the range of sizes stay the same? Would your method still give the right answer?

**Turn 2 [student]**

> I think if I cut all the toys in half, the range might stay the same because everything is getting smaller by the same amount. But I'm not sure about adding the same amount, like if I added \( 3 \mathrm{~cm} \) to each toy. Would that change the range? I’m confused about how that works.

**Turn 3 [teacher]**

> Imagine you had a group of toys that you were comparing in terms of their sizes. If you added \(3 \mathrm{~cm}\) to each toy, would the range stay the same? Would your method still give the right answer?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses generic analogies without directly addressing the proportional vs. additive misconception, and the student's final turn shows they still believe cutting in half keeps the range the same while remaining confused about addition, which is the opposite of the correct understanding.

**Pass 2 (R6, persona-visible):** The teacher consistently uses a concrete, everyday analogy (toys and their sizes) to help the student understand the abstract concept of range under transformations, which is a strong implementation of the experiential strategy.

---

## case_2 / experiential / baseline / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine if you had a collection of toys, and you cut them all in half. Would the range of sizes stay the same?

**Turn 2 [student]**

> Hmm, well if I cut them all in half, they'd all be smaller... so like, if one toy was 10 cm and another was 20 cm, after cutting they'd be 5 cm and 10 cm. Oh wait, the difference between them would still be the same wouldn't it? So maybe the range wouldn't change when you cut things in half?

**Turn 3 [teacher]**

> Imagine now if you had a collection of toys and you added 3 cm to each toy's height. Would the range of sizes stay the same?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher's approach leads the student to adopt an incorrect belief (that cutting in half preserves range), which is mathematically wrong and opposite to the student's initial correct answer, showing poor misconception engagement and domain accuracy.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, everyday analogies (a collection of toys) to help the student understand the abstract mathematical concept of range, which is a textbook example of the experiential strategy.

---

## case_2 / experiential / hybrid / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you have a basket of apples, each with a different weight. If you cut all the apples in half, would the heaviest apple still be the heaviest? Think about how changing the values by the same proportion can affect the range in this scenario.

**Turn 2 [student]**

> I think I get what you're saying about the apples. If I cut them in half, the heaviest apple would still be the heaviest, so the range might stay the same. But what about when Katie says if all the plants grow by \( 3 \mathrm{~cm} \)? Wouldn't that change the range too? I'm still a bit confused about how that works.

**Turn 3 [teacher]**

> Think of it like this: imagine you have a group of friends, each with a different height. If they all grew by the same amount, would the tallest friend still be the tallest? Consider how changing the values by a constant amount affects the range in this situation.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher's apple analogy in Turn 1 is mathematically incorrect (cutting in half does change the range), reinforcing rather than correcting the misconception, and by Turn 2 the student shows increased confusion about both scenarios.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, everyday analogies (basket of apples with different weights, group of friends with different heights) to help the student understand abstract mathematical concepts about range and transformations, which is a textbook example of the experiential strategy.

---

## case_2 / experiential / hybrid / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this in terms of changing the heights of the plants. Remember how in the past, we discussed how changing all the heights by the same amount doesn't necessarily keep the range the same? Let's apply that same idea to this situation with the plants being cut in half or growing by \(3 \mathrm{~cm}\) each.

**Turn 2 [student]**

> Oh right, so you're saying that adding the same amount to all the heights changes the range? But I thought... um, doesn't cutting them in half change things differently than just adding 3 cm? Like, I'm confused about why one would change the range and the other wouldn't - can you explain that bit more?

**Turn 3 [teacher]**

> Let's think of it this way: imagine you have a group of friends, and you all decide to either double your allowance or increase it by $5 each. Would the range of everyone's allowance stay the same in both scenarios? Think about how changing all the heights by the same proportion might affect the range in a similar way.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher misrepresents the mathematical content in Turn 1 (claiming adding the same amount doesn't keep range the same, when it does), confuses the student further, and by Turn 2 the student is more confused than initially, showing no improvement in understanding the actual misconception about proportional changes.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the experiential strategy by grounding the abstract mathematical concept of range in a concrete, relatable real-world analogy about friends and allowances, making the proportional versus additive change distinction tangible through everyday experience.

---

## case_2 / experiential / pure_ai / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about it in terms of a group of friends playing basketball. If everyone in the group grew taller by the same amount, would the tallest and shortest person in the group stay the same? Or would the range of heights change?

**Turn 2 [student]**

> I think if everyone grew taller by the same amount, the tallest and shortest person would still be the same, so the range wouldn't change. But I’m not sure if that’s right because I thought the range only depends on the numbers. Does it really change if they all grow by the same amount?

**Turn 3 [teacher]**

> Let's imagine you and your friends are all given a dollar each. If everyone in the group earned double the money they started with, would the person with the most money and the person with the least money still be the same? Think about how this scenario relates to changing all the plant heights by the same amount.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses analogies that confuse additive vs. multiplicative changes (the basketball analogy addresses addition correctly, but the money analogy conflates doubling with the original problem), and the student shows no improvement—they correctly understood Katie's case initially but now express confusion about whether adding the same amount changes the range.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, relatable real-world analogies (basketball players' heights, friends earning money) to help the student understand the abstract mathematical concept of range, which is a textbook example of the experiential strategy.

---

## case_2 / experiential / pure_ai / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about it this way: Imagine you have a group of friends, and you all decide to either double your pocket money or add $5 to it. Would everyone end up with the same amount of money as before, or would the range of money each person has change?

**Turn 2 [student]**

> Hmm, so if we all doubled our pocket money, everyone would have more, so the range would change... but if we all added £5, then like, the difference between the richest and poorest person would stay the same, right? So Katie would be correct then? I'm confused about why Tom would be right if we're cutting the plants in half though.

**Turn 3 [teacher]**

> Let's imagine instead of cutting the plants in half, we double the height of each plant. Would the range of heights stay the same or change? Think about how the tallest and shortest plants would be affected by this change.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively addresses the proportional change misconception through well-designed prompts requiring reasoning, and the student makes partial progress (correctly understanding addition but still confused about multiplication), though the dialogue ends before full resolution.

**Pass 2 (R6, persona-visible):** The teacher consistently translates the abstract mathematical concept of range under transformations into concrete, relatable scenarios (pocket money among friends, plant heights) that ground the explanation in everyday experiences students can visualize and reason about.

---

## case_2 / experiential / pure_cbr_llm / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this in terms of a different scenario. Imagine you have a group of friends, and you all go on a hike together. If everyone walks at half their usual speed, do you think the time it takes to finish the hike would change? Think about how this relates to Tom and Katie's discussion about the heights of the plants.

**Turn 2 [student]**

> I think if everyone walks at half their usual speed, it would take longer to finish the hike. So maybe that means if you cut the plant heights in half, the range would change too? But I’m still not sure how that works with Katie’s idea about adding \( 3 \mathrm{~cm} \). Can you explain that part more?

**Turn 3 [teacher]**

> Imagine you have a recipe for a cake that calls for 2 cups of flour. If you double the recipe, how much flour would you need? Think about how this relates to Tom and Katie's discussion about the plant heights.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses analogies that don't clearly map to the range concept (hiking speed doesn't preserve the analogy structure, and the recipe analogy is introduced when the student is confused), the student shows increased confusion rather than progress, and the specific misconception about proportional changes versus additive changes affecting range is never directly addressed.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (hiking, baking), these analogies are poorly matched to the mathematical concept of range and actually confuse rather than clarify the relationship between scaling/shifting operations and range, as evidenced by the student's increased confusion in Turn 2.

---

## case_2 / experiential / pure_cbr_llm / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about it like this: if you have a group of friends and you all grow by 3 cm, would everyone still be in the same order from shortest to tallest? Think about how that relates to the range of the plant heights.

**Turn 2 [student]**

> Yeah, I guess everyone would still be in the same order... so like, if the shortest plant is still the shortest and the tallest is still the tallest, then the difference between them would be the same? But wait, I'm confused because I thought adding the same amount to everything wouldn't change the range, so why did I say only Tom was right?

**Turn 3 [teacher]**

> Let's think back to the misconception about a lower average meaning the data is more consistent. Just like how a lower average doesn't necessarily mean the data is more consistent, adding the same amount to every plant may not necessarily keep the range the same. Let's explore this idea further together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher introduces an irrelevant misconception about averages in Turn 3, provides mathematically incorrect guidance suggesting adding a constant might change the range, and fails to help the student recognize that Katie (not Tom) is correct, leaving the student more confused than at the start.

**Pass 2 (R6, persona-visible):** While Turn 1 uses a concrete analogy about friends growing taller, Turn 3 abandons the experiential approach entirely by referencing an unrelated abstract misconception about averages rather than continuing with real-world grounding.

---

## case_2 / experiential / pure_cbr_tpl / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get it a bit better now, but I'm still confused. So, if Tom cuts all the plants in half, the range would actually change, right? And what about Katie's idea of adding \( 3 \mathrm{~cm} \)? Does that change the range too?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions rather than addressing the student's actual confusion about how multiplication and addition affect range, resulting in no meaningful engagement or progress.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or experiential grounding whatsoever, instead repeatedly copying generic diagnostic information about unrelated misconceptions that doesn't address the student's question about plant heights and range.

---

## case_2 / experiential / pure_cbr_tpl / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Um, okay... so I think I need to actually work it out properly then? Like, should I calculate what the range actually is first with the original numbers, and then see what happens when I cut them in half or add 3 cm to each one?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions rather than addressing the student's actual confusion about how range changes under different transformations, resulting in no meaningful engagement or progress.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations whatsoever, instead repeatedly copying generic misconception examples that are unrelated to the student's plant height problem.

---

## case_2 / rule_based / baseline / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, changing all values by the same proportion will affect the range. Let's work through the calculations together to see how the range changes when the plants are cut in half or grow by \(3 \mathrm{~cm}\) each.

**Turn 2 [student]**

> Okay, I think I get that we need to calculate the range for both situations. So, if we cut the heights in half, we would have to find the new highest and lowest values, right? And then for the \(3 \mathrm{~cm}\) increase, we do the same? I'm just not sure how that will change the range. Can you help me with the calculations?

**Turn 3 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, changing all values by the same proportion will affect the range. Let's work through the calculations together to see how the range changes when the plants are cut in half or grow by \(3 \mathrm{~cm}\) each.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeats a generic procedural statement without engaging the student's specific confusion, provides no actual scaffolding for calculations the student requests, and the student shows no improvement in understanding by the end (the teacher's Turn 3 is identical to Turn 1, suggesting complete lack of responsiveness).

**Pass 2 (R6, persona-visible):** While the teacher mentions a procedure and rules, they fail to actually show worked steps, provide explicit calculations, or give direct corrective feedback on the student's specific error, instead repeating vague statements about procedures without demonstrating them.

---

## case_2 / rule_based / baseline / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, if all values are changed by the same proportion, the range will indeed change.

**Turn 2 [student]**

> Oh right, so you're saying the range *does* change when you multiply by the same proportion? But I thought Tom was right because... um, I'm a bit confused now. Can you show me what happens to the range when we actually cut all the plants in half?

**Turn 3 [teacher]**

> There is a specific procedure for this. Step 1 is to identify the operation; step 2 is to apply the rule for that operation. In this case, when we cut all the plants in half, the range will change. Let's work through the calculation together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies the mathematical rule but merely states it repeatedly without engaging the student's reasoning, and the student remains confused throughout, showing no improvement in understanding.

**Pass 2 (R6, persona-visible):** While the teacher states there is a 'specific procedure' and mentions steps, they never actually show the worked steps or calculations, nor do they provide the explicit procedural rules for how operations affect range, which are core requirements of rule-based instruction.

---

## case_2 / rule_based / hybrid / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> The range of a set of data is affected by the difference between the largest and smallest values. Even if all values are changed by the same proportion, the range can still change. Let's walk through the calculation together to see where the misconception occurred.

**Turn 2 [student]**

> Okay, I think I get that the range is the difference between the biggest and smallest numbers. So if I cut all the heights in half, wouldn't the biggest and smallest numbers also change, but still keep the same difference? I'm not sure how that works with Katie's example either. Can you explain that part again?

**Turn 3 [teacher]**

> When all values are changed by the same proportion, the actual numerical difference between the largest and smallest values can still change, affecting the range. Let's work through an example to illustrate this concept further.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher identifies the misconception but repeatedly tells rather than guides discovery, provides mathematically incorrect statements (claiming proportional changes affect range when only multiplicative changes do), and the student remains confused by Turn 2, showing no improvement in understanding.

**Pass 2 (R6, persona-visible):** The teacher discusses concepts about range and proportional changes but never states explicit procedures, shows worked calculation steps, or gives direct corrective feedback on the student's specific procedural error of confusing additive versus multiplicative transformations.

---

## case_2 / rule_based / hybrid / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To determine if changing all values by the same proportion affects the range, remember that the range is the difference between the highest and lowest values. Let's walk through the calculation together step by step. This will help clarify why changing the values by the same proportion does indeed affect the range.

**Turn 2 [student]**

> Okay, so the range is the highest minus the lowest... so for the plants that would be 42 minus 13 which is 29 cm. But I'm still confused about why cutting them in half would change it - wouldn't 21 minus 6.5 still be the same difference?

**Turn 3 [teacher]**

> When you cut all the plant heights in half, you're reducing each height by the same proportion. This will affect the range because the difference between the highest and lowest values will also be halved. So, the new range would be \( 21 \mathrm{~cm} - 6.5 \mathrm{~cm} = 14.5 \mathrm{~cm} \), which is different from the original range of 29 cm. Remember, changing values by the same proportion does impact the range.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies and addresses the specific misconception about proportional changes, but primarily tells rather than elicits reasoning, and the student shows no evidence of improved understanding by the end (never gets to speak after the final explanation).

**Pass 2 (R6, persona-visible):** The teacher consistently provides explicit procedural steps (calculating range as highest minus lowest), shows worked calculations with specific numbers (21 - 6.5 = 14.5 cm), and gives direct corrective feedback on the student's procedural error about proportional changes affecting range.

---

## case_2 / rule_based / pure_ai / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To determine if the range changes when all values are changed by the same proportion, we must consider how the values spread out from each other. Let's revisit the definition of range and how changing all values affects it. Can you explain how the range is calculated and what changing all values by the same proportion does to the spread of the values?

**Turn 2 [student]**

> I think the range is found by taking the biggest number and subtracting the smallest number. So, if we cut all the plants in half, the biggest and smallest would both be smaller, but I thought they would still be the same distance apart. But if we add \( 3 \mathrm{~cm} \) to each, wouldn't that just make everything bigger but not change how far apart they are? I'm still a bit confused about how the range actually changes.

**Turn 3 [teacher]**

> When all values are changed by the same proportion, the spread of the values changes, affecting the range. Cutting all the plants in half would indeed change the range, as the difference between the smallest and largest values would also decrease. Adding 3 cm to each plant would also impact the range by increasing the spread of the values. The range is not just about the distance between the smallest and largest values, but also how the values are distributed within that range.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming adding a constant changes the range and misdefining what range measures), fails to effectively address the student's misconception about proportional changes, and the student remains confused by the end with no visible improvement in understanding.

**Pass 2 (R6, persona-visible):** The teacher uses inquiry-based questioning rather than stating explicit procedures or showing worked steps, and provides conceptually incorrect information instead of direct corrective feedback on the procedural error.

---

## case_2 / rule_based / pure_ai / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To determine if the range changes when all values are changed by the same proportion, we need to consider the definition of range. The range is the difference between the highest and lowest values. If all values are cut in half or increased by the same amount, the range will indeed change. Let's revisit the calculation process to clarify this concept.

**Turn 2 [student]**

> Oh right, so you're saying the range *does* change in both cases? So like, when you cut all the plants in half, the difference between the biggest and smallest also gets cut in half? But I thought if you're just making everything smaller by the same amount, the gap between them would stay the same... can you show me with actual numbers so I can see what's happening?

**Turn 3 [teacher]**

> Yes, that's correct! Let's work through an example with actual numbers to see how the range changes when all values are cut in half. I'll show you step by step. Let's start by calculating the range of the original plant heights.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes a critical mathematical error by stating that the range changes when values are increased by the same amount (it doesn't), confusing the student who initially had the correct answer, and the student's final turn shows they've been led away from correct reasoning rather than toward it.

**Pass 2 (R6, persona-visible):** While the teacher mentions showing steps and working through an example, they provide conceptual explanations without actually stating explicit procedures, showing worked calculations, or giving direct corrective feedback on the student's specific procedural error of concluding 'Only Tom.'

---

## case_2 / rule_based / pure_cbr_llm / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To determine if the range would change if all the plants were cut in half or grew by 3 cm each, remember that the range is the difference between the highest and lowest values in the data set. Think about how changing all values by the same proportion would affect the spread of the heights. Can you identify where the misconception may lie in your approach?

**Turn 2 [student]**

> I think I get what you're saying about the range being the difference between the highest and lowest values. But I'm still confused about how cutting the plants in half or adding 3 cm would change that. I thought if you just changed all the numbers by the same amount, it wouldn't really affect the range. Can you explain more about why that isn't true?

**Turn 3 [teacher]**

> Changing all values by the same proportion can affect the range, as seen in the misconception where a difference in one part of a ratio does not necessarily result in a difference of one unit. Let's walk through an example to see how changing all values by the same amount can impact the spread of the data.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to clearly distinguish between proportional changes (cutting in half) and additive changes (adding 3cm), provides confusing and inaccurate explanations about ratios, and the student remains confused by Turn 2, showing no improvement in understanding the actual misconception.

**Pass 2 (R6, persona-visible):** The teacher uses questioning and hints to guide discovery rather than stating explicit procedures, showing worked steps, or giving direct corrective feedback on the procedural error, which are hallmarks of rule-based instruction.

---

## case_2 / rule_based / pure_cbr_llm / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To find the range, you need to subtract the smallest value from the largest value. In this case, if all the plants were cut in half or grew by the same amount, the range would indeed change. So, the range does change when values are changed by the same proportion. Remember, range is affected by the actual values, not just the proportion.

**Turn 2 [student]**

> Oh right, so you're saying the range *does* change in both cases? But I thought if you cut them all in half, like the difference between them stays the same... can you show me with actual numbers so I can see what you mean?

**Turn 3 [teacher]**

> When you cut the heights of the plants in half, the new heights would be \( 12 \mathrm{~cm}, 8.5 \mathrm{~cm}, 21 \mathrm{~cm}, 13 \mathrm{~cm}, 6.5 \mathrm{~cm} \). Calculate the new range by subtracting the smallest from the largest value to see how the range changes. Remember, changing the values by the same proportion will affect the range.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming range changes with multiplication, when it actually scales proportionally), confuses the student who initially had the correct answer, and fails to address the core misconception about proportional versus additive changes.

**Pass 2 (R6, persona-visible):** While the teacher provides some worked steps in Turn 3 by showing the halved values, the explanation in Turn 1 contains a significant mathematical error (incorrectly stating that cutting in half changes the range) and lacks the clear procedural guidance and direct corrective feedback characteristic of rule-based instruction.

---

## case_2 / rule_based / pure_cbr_tpl / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Okay, I think I see where I might have gone wrong. So, if Tom cuts the plants in half, the heights would change, but the range might still change too, right? And for Katie, if all the plants grow by \( 3 \mathrm{~cm} \), that would also change the range. Can you help me understand how the range actually changes in both cases?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions rather than addressing the student's actual confusion about how range changes under multiplication versus addition, making this dialogue completely ineffective.

**Pass 2 (R6, persona-visible):** The teacher provides no explicit procedures, worked steps, or direct corrective feedback, instead repeatedly presenting irrelevant example problems that don't address the student's question about how range changes under transformations.

---

## case_2 / rule_based / pure_cbr_tpl / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Um, okay... so I think I need to actually work it out properly then? Like, should I calculate what the range actually is first with the original numbers, and then see what happens when I cut them in half or add 3 cm to each one?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples about unrelated misconceptions (range units, mean vs range, ratios) that don't address the student's actual misconception about proportional changes, and the student makes no progress toward understanding why Katie is correct.

**Pass 2 (R6, persona-visible):** The teacher never states explicit procedures, shows worked steps, or gives direct corrective feedback; instead, they repeatedly provide irrelevant examples from unrelated problems that don't address the student's range calculation question.

---

## case_2 / socratic / baseline / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me ask you this: what would happen if you tried your method on a simpler version of the problem?

**Turn 2 [student]**

> Okay, I guess I could try it with just two plants, like \( 10 \mathrm{~cm} \) and \( 20 \mathrm{~cm} \). If I cut them in half, they'd be \( 5 \mathrm{~cm} \) and \( 10 \mathrm{~cm} \), so the range would still be \( 5 \mathrm{~cm} \). But if I added \( 3 \mathrm{~cm} \) to both, they'd be \( 13 \mathrm{~cm} \) and \( 23 \mathrm{~cm} \), and the range would still be \( 10 \mathrm{~cm} \). I’m not sure why that works differently for Katie's idea though. Can you help me understand that?

**Turn 3 [teacher]**

> Let me ask you this: why do you think the range changed when you increased the heights by \( 3 \mathrm{~cm} \) but stayed the same when you cut them in half?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to address that the student's calculation in Turn 2 is actually correct (both operations preserve the range of 10), missing the real misconception, and the final question reinforces a false premise that the range changed when it didn't.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without giving direct answers, perfectly exemplifying the Socratic method by having the student work through a simpler case and then questioning their interpretation of the results.

---

## case_2 / socratic / baseline / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me ask you this: what would happen if you tried your method on a simpler version of the problem?

**Turn 2 [student]**

> Um, okay... so like, if I had simpler numbers, maybe 10 and 20? So the range would be 10. Then if I cut them in half I'd get 5 and 10, and the range would still be 5... wait, that changed though, didn't it? But if I add 3 to both of them I'd get 13 and 23, and that's still a range of 10... so actually maybe both of them could be right? I'm confused now.

**Turn 3 [teacher]**

> It seems like you're starting to see how changing all values by the same proportion can affect the range differently. Can you think of another way to test this idea with different numbers?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 5 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively addresses the proportional change misconception through well-calibrated prompts that lead the student to discover the error independently, resulting in clear conceptual progress from believing only Tom is right to recognizing that Katie (not Tom) is correct.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide discovery, never directly reveals the answer, and successfully helps the student recognize their flawed reasoning through guided exploration with simpler examples.

---

## case_2 / socratic / hybrid / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you explain why you think the range wouldn't change if all the plants were cut in half or grew by \(3 \mathrm{~cm}\) each? Let's consider how changing all values by the same proportion affects the spread of the data.

**Turn 2 [student]**

> I thought that if you cut all the heights in half, the biggest and smallest plants would still be the same distance apart, so the range wouldn't change. And with Katie's example, if you just add \(3 \mathrm{~cm}\) to each height, it seems like the difference between the tallest and shortest would still be the same. But now I'm not sure if that's right. Can you help me understand how the range actually changes?

**Turn 3 [teacher]**

> Can you think of a scenario where changing all values by the same proportion does affect the range? Consider how the relationship between the heights of the plants might change if they were all doubled or tripled in size. How would that impact the spread of the data?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to address the student's actual misconception (confusing proportion with addition), provides mathematically confusing guidance by conflating these concepts, and the student shows no improvement—remaining confused at the end without resolving their misunderstanding about which operation preserves range.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking and never directly provides the answer, though they could have more effectively used a counterexample to expose the student's flawed reasoning about cutting in half.

---

## case_2 / socratic / hybrid / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's explore the impact of changing all values by the same proportion on the range. Can you think of a scenario where changing all values by the same proportion would result in a different range?

**Turn 2 [student]**

> Um, I'm not really sure... like, if we cut all the plants in half, wouldn't the difference between the tallest and shortest stay the same because we're doing the same thing to all of them? But I'm confused about what you mean by "proportion" - is that different from just adding or subtracting the same amount?

**Turn 3 [teacher]**

> Let's consider the misconception that changing all values by the same proportion would not affect the range. Can you think of a situation where changing all values by the same proportion would actually result in a different range?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly asks the same abstract question without addressing the student's confusion about proportion vs. addition or providing concrete guidance, resulting in no visible progress in the student's understanding of their misconception.

**Pass 2 (R6, persona-visible):** The teacher uses probing questions and avoids giving direct answers, which aligns with Socratic method, but repeats essentially the same question without adapting to the student's confusion about 'proportion' versus addition, and doesn't use counterexamples to expose the flawed reasoning as the strategy prescribes.

---

## case_2 / socratic / pure_ai / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you explain why you think changing all values by the same proportion wouldn't affect the range? What happens to the differences between the heights of the plants when they are cut in half or grow by the same amount?

**Turn 2 [student]**

> I thought that if you cut all the heights in half, the differences between them would stay the same, so the range wouldn't change. But now I'm not sure if that's right. When you grow them by \( 3 \mathrm{~cm} \), does that change the differences too? Can you help me understand how the range works with those changes?

**Turn 3 [teacher]**

> You're on the right track in recognizing that changing the values by the same proportion can affect the differences between them. Can you think of a specific example with numbers to see how cutting the heights in half or growing them by \( 3 \mathrm{~cm} \) might change the range?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively engages the misconception and maintains cognitive demand through questioning, but makes a mathematical error in Turn 3 by suggesting proportion changes 'can affect' differences (when halving actually does change the range), and the student shows confusion rather than clear progress by Turn 2.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking and never directly provides the answer, though they could have more explicitly used a counterexample to expose the flawed reasoning about proportional changes.

---

## case_2 / socratic / pure_ai / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this further. Can you explain why you believe the range wouldn't change if all the plants were cut in half or grew by the same amount? How does changing all values by the same proportion affect the spread of the data set?

**Turn 2 [student]**

> Um, I think... when you cut all the plants in half, the difference between the biggest and smallest plant gets smaller too, so the range changes. But when you add 3 cm to all of them, the gap between them stays the same, so the range doesn't change? Is that right, or am I still getting it wrong?

**Turn 3 [teacher]**

> You're on the right track! Can you think of a scenario where changing all values by the same proportion does affect the range? How might this apply to the plants' heights in this situation?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** While the student successfully corrects their misconception by Turn 2, the teacher's Turn 3 response is mathematically confusing (asking them to find a scenario where proportional change affects range, when the student just correctly explained it does) and fails to affirm the correct reasoning the student has already demonstrated.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without giving direct answers, though they could have used a counterexample to more explicitly expose the flaw in the student's initial reasoning about Tom being correct.

---

## case_2 / socratic / pure_cbr_llm / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you explain why you think the range wouldn't change if all the plants were cut in half or grew by 3 cm each? What about the concept of range leads you to believe that?

**Turn 2 [student]**

> I thought that if you cut all the plants in half, the biggest and smallest would still be the same distance apart, so the range wouldn't change. And for Katie, I thought adding the same amount to each height wouldn't change the difference between the tallest and shortest either. But now I'm not sure if that’s really right. Can you help me understand how the range actually works?

**Turn 3 [teacher]**

> Can you think of any examples where changing all values by the same proportion would actually change the range?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher's final question contains a critical mathematical error (asking about proportional change when the student correctly reasoned about it in Turn 2), fails to acknowledge that the student actually had the right reasoning for Katie's case, and provides no evidence that the student's understanding improved since they never get to respond to the flawed prompt.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking (asking about their reasoning in Turn 1, prompting them to consider counterexamples in Turn 3) and never directly provides the answer, which is textbook Socratic method.

---

## case_2 / socratic / pure_cbr_llm / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you explain why you think changing all the plant heights by the same proportion wouldn't affect the range?

**Turn 2 [student]**

> Um, I think it's because like... if you cut them all in half, the difference between the biggest and smallest plant stays the same? Like the gap between them doesn't change even though the numbers get smaller. But I'm not really sure why adding the same amount to all of them would be different...

**Turn 3 [teacher]**

> Can you think of a situation where changing all the values by the same proportion would actually change the range?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly engages the misconception and maintains high cognitive demand, but the scaffolding in Turn 3 asks the student to find a counterexample when they've just revealed confusion about the core concept, and there's no evidence the student's understanding has improved by the dialogue's end.

**Pass 2 (R6, persona-visible):** While the teacher uses questions rather than direct answers, the question in Turn 3 actually contradicts the student's correct intuition (halving does change the range), which undermines rather than guides the student's reasoning through proper Socratic questioning.

---

## case_2 / socratic / pure_cbr_tpl / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Okay, I think I get that I was wrong about Katie too. If all the plants grow by \( 3 \mathrm{~cm} \), that would change the heights, right? But I'm still not sure how cutting them in half affects the range. Can you help me understand that part better?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions rather than addressing the student's actual confusion about proportional changes and range, making no meaningful pedagogical contribution to the dialogue.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions or use counterexamples; instead, they repeatedly paste irrelevant information about misconceptions from unrelated problems, completely failing to engage in Socratic dialogue.

---

## case_2 / socratic / pure_cbr_tpl / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Yeah, I think I need to work through it properly because I'm not really sure what happens to the range when you change the numbers. Like, does the range stay the same no matter what you do to all the plants, or does it actually change? Can you show me how to check both Tom's and Katie's ideas?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples from unrelated topics instead of addressing the student's misconception about how operations affect range, making no progress toward helping the student understand that multiplicative changes affect range while additive changes don't.

**Pass 2 (R6, persona-visible):** The teacher completely fails to use the Socratic method, instead repeatedly pasting irrelevant example problems rather than asking probing questions to guide the student's thinking about how operations affect range.

---

## case_2 / traditional / baseline / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me explain the correct method, and then we will work through an example together. In this case, changing all values by the same proportion can indeed affect the range. Let's go through a worked example to help clarify this concept.

**Turn 2 [student]**

> Okay, I think I get that changing all the values can affect the range, but I'm still a bit confused. So, if we cut all the heights in half, how would that change the range? And what about if we add \( 3 \mathrm{~cm} \) to each height? Can you show me how to work it out?

**Turn 3 [teacher]**

> Let's work through these examples together to see how changing the values affects the range. Let's start by cutting all the heights in half and then adding \(3 \mathrm{~cm}\) to each height. This will help clarify the concept for you.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly promises to work through examples but never actually does so, leaving the student without concrete calculations to address their misconception about proportional versus additive changes to range.

**Pass 2 (R6, persona-visible):** While the teacher mentions working through examples and checking understanding, they repeatedly promise to explain and work through examples without actually providing the direct explanations, worked examples, or clear step-by-step solutions that are central to the traditional strategy.

---

## case_2 / traditional / baseline / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me explain the correct method, and then we will work through an example together. If all the plants were cut in half or grew by the same amount, the range would actually change. Let's calculate the new range together.

**Turn 2 [student]**

> Oh right, so you're saying the range *does* change in both cases? That's confusing because I thought if you change all the numbers by the same amount, like cutting them all in half or adding 3 to each one, the difference between the biggest and smallest would stay the same... Can you show me how to work it out step by step?

**Turn 3 [teacher]**

> Let me explain the correct method, and then we will work through an example together. Changing all the values by the same proportion does affect the range. Let's calculate the new range step by step.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming Katie is wrong when she's actually right), never engages with the student's specific reasoning about proportional vs. additive changes, demands no cognitive work from the student, and the student ends more confused than at the start.

**Pass 2 (R6, persona-visible):** The teacher repeatedly makes incorrect mathematical claims (that adding a constant changes the range, which is false) and fails to provide any worked examples or clear explanations despite promising to do so twice, which contradicts the core elements of traditional teaching.

---

## case_2 / traditional / hybrid / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's revisit the concept of range and how it is affected by changes in values. Remember, the range is the difference between the highest and lowest values in a set. Think about how cutting all the plants in half or growing them by the same amount would impact the range. Can you see how the range may change in each scenario?

**Turn 2 [student]**

> I think I get that the range is the difference between the highest and lowest values. So if we cut all the plants in half, the highest and lowest would also be cut in half, right? But I'm still not sure how that affects the range. Does it stay the same? And with Katie's idea, if we just add \( 3 \mathrm{~cm} \) to each plant, would that change the range too?

**Turn 3 [teacher]**

> Let's work through an example to clarify this concept. Imagine we have a set of numbers: 2, 4, 6, 8, 10. If we cut each number in half, what would the new range be? Let's calculate it together to see how changing all values by the same proportion affects the range.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses the misconception by introducing a worked example to distinguish between proportional and additive changes, but the student shows no clear resolution by Turn 2 and the dialogue ends before demonstrating improved understanding.

**Pass 2 (R6, persona-visible):** The teacher uses questioning and guided discovery rather than direct explanation with worked examples, which contradicts the traditional strategy of explaining directly and providing complete worked examples.

---

## case_2 / traditional / hybrid / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's focus on the misconception that changing all values by the same proportion wouldn't affect the range. Think about how changing all the plant heights by the same amount would impact the difference between the tallest and shortest plant. Can you see how the range would actually change in this scenario?

**Turn 2 [student]**

> Hmm, so when Katie adds 3 cm to all of them, the difference between the tallest and shortest would still be the same because you're adding the same amount to both? But then why would Tom be right about cutting them in half - wouldn't that change the range too if we're dividing everything by the same number?

**Turn 3 [teacher]**

> You're on the right track with the idea of adding the same amount to each height not changing the range. When cutting the heights in half, think about how the ratios of the heights would change and how that would affect the range. Can you see how dividing by the same number would impact the range differently than adding the same amount?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the proportional change misconception and the student shows clear progress by Turn 2 in understanding that adding maintains range while dividing changes it, though the teacher's scaffolding could have been more concrete given the student's initial confusion.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning to lead the student to discover the answer rather than directly explaining with worked examples, which is inconsistent with the traditional strategy of explicit instruction.

---

## case_2 / traditional / pure_ai / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's revisit the concept of range in this context. Remember, the range is the difference between the highest and lowest values in a set of data. Think about how cutting all the plant heights in half or adding the same amount to each height would affect the overall range. Can you explain your reasoning for why you agree with Tom?

**Turn 2 [student]**

> I thought that if you cut all the heights in half, the highest and lowest would still be in the same proportion, so the range wouldn't change. But now I'm not sure if that's right. Can you explain how adding \( 3 \mathrm{~cm} \) would change the range?

**Turn 3 [teacher]**

> Let's consider how adding \( 3 \mathrm{~cm} \) to each plant height would affect the range. Remember, the range is the difference between the highest and lowest values. Think about how increasing each height by the same amount would impact the overall range. Can you describe your thought process as you work through this scenario?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly prompts for reasoning without directly addressing the student's specific misconception about proportional changes, and the student shows no improvement—actually becoming more confused by Turn 2 and not responding substantively by Turn 3.

**Pass 2 (R6, persona-visible):** The teacher consistently uses questioning and prompting to elicit student reasoning rather than providing direct explanations or worked examples, which is contrary to the traditional strategy of explicit instruction.

---

## case_2 / traditional / pure_ai / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's revisit the concept of range in this context. Remember, the range is the difference between the highest and lowest values in a set of data. Think about how cutting all the plant heights in half or adding \(3 \mathrm{~cm}\) to each height would affect the range. Can you explain your reasoning behind your choice of Tom as the one you agree with?

**Turn 2 [student]**

> Um, so I thought Tom was right because if you cut all the plants in half, the range stays the same... but now I'm confused because if you add 3 cm to each plant, wouldn't that also keep the range the same since you're doing the same thing to all of them? Like, shouldn't both of them be right then?

**Turn 3 [teacher]**

> Let's think about it this way: when you cut all the plants in half, the difference between the highest and lowest heights will still be the same because you're reducing all the values by the same proportion. However, when you add 3 cm to each height, you're actually increasing the range because the difference between the highest and lowest heights will now be greater. Can you see why Katie's reasoning makes more sense in this scenario?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming proportional changes preserve range while additive changes increase it, when the opposite is true), fails to engage the student's emerging correct reasoning in Turn 2, and the student shows no improvement by the end.

**Pass 2 (R6, persona-visible):** The teacher uses questioning and guided discovery rather than direct explanation with worked examples, and provides incorrect mathematical information (claiming adding a constant increases the range, when it actually keeps it constant), which contradicts the traditional strategy's emphasis on clear, direct instruction.

---

## case_2 / traditional / pure_cbr_llm / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's focus on the misconception that changing all values by the same proportion would not affect the range. Can you explain why you think Tom's statement may or may not be correct, based on what we know about range calculations?

**Turn 2 [student]**

> I think Tom's statement might be correct because if you cut all the heights in half, they all get smaller but maybe the difference between the tallest and shortest stays the same? I'm not really sure how that works with the numbers. Can you help me understand why the range changes or doesn't change?

**Turn 3 [teacher]**

> Let's work through an example to help clarify this concept. Let's consider the heights of the plants if they were all cut in half. Can you calculate the new range for the heights of the plants after they have been cut in half?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the proportional change misconception and asks the student to calculate through an example, but the student's final turn shows they haven't yet resolved their confusion, only that they're willing to work through the calculation.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning rather than direct explanation with worked examples, which is inconsistent with the traditional strategy of explicitly explaining concepts and demonstrating solutions before checking understanding.

---

## case_2 / traditional / pure_cbr_llm / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's first clarify the concept of range and how changing all values by the same proportion can affect it. Can you explain to me why you think Tom's statement is correct and how you approached working it out?

**Turn 2 [student]**

> Um, I thought that if you cut all the plants in half, like divide them all by 2, then the difference between the biggest and smallest would stay the same because you're doing the same thing to all of them? But now I'm not really sure... does it actually change the range or not?

**Turn 3 [teacher]**

> Let's work through a quick example to clarify this. If we have the heights of the plants as \( 24, 17, 42, 26, 13 \) and we cut them all in half, what would the new heights be? Then, we can compare the new range to the original range to see if it changes.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly addresses the proportional change misconception and provides appropriate scaffolding through concrete calculation, but the dialogue ends before we can observe whether the student actually corrects their understanding about how multiplication affects range.

**Pass 2 (R6, persona-visible):** The teacher asks probing questions to elicit student thinking rather than directly explaining the concept with worked examples as the traditional strategy prescribes, though Turn 3 does begin to move toward a worked example approach.

---

## case_2 / traditional / pure_cbr_tpl / leg_a

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get what you're saying, but I'm still a bit confused. So, if Tom cuts all the heights in half, the range does change, right? And for Katie, if all the plants grow by 3 cm, the range changes too? Can you explain why that happens?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions instead of addressing the student's confusion about how transformations affect range, resulting in no meaningful engagement, no cognitive demand, no appropriate scaffolding, questionable domain accuracy in the examples given, and no student progress (the student remains confused and asks for explanation in Turn 2 but receives the same unhelpful response).

**Pass 2 (R6, persona-visible):** The teacher provides no direct explanation or worked examples, instead repeatedly copying irrelevant misconception examples that don't address the student's question about how range changes with multiplication versus addition.

---

## case_2 / traditional / pure_cbr_tpl / leg_b

**Misconception:** Believes if you changed all values by the same proportion the range would not change

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: Tom and Katie are discussing the \( 5 \) plants with these heights: \( 24 \mathrm{~cm}, 17 \mathrm{~cm}, 42 \mathrm{~cm}, 26 \mathrm{~cm}, 13 \mathrm{~cm} \) Tom says if all the plants were cut in half, the range wouldn't change. Katie says if all the plants grew by \( 3 \mathrm{~cm} \) each, the range wouldn't change. Who do you agree with? I got Only
> Tom because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Um, okay... so I think I need to actually work it out properly then? Like, should I calculate what the range actually is first with the original numbers, and then see what happens when I cut them in half or add 3 cm to each one?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Calculate the range from a list of data. Correct answer (C): Add all the numbers together.. Student selected: Check the units of your answer. Underlying misconception: Believes range does not need units. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Compare sets of data using the range. Correct answer (A): The girls because their range is lower. Student selected: The boys because their mean is lower. Underlying misconception: Thinks a lower average means the data is more consistent. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Given information about one part, work out other parts. Correct answer (D): \( 1600 \ \mathrm{ml} \). Student selected: \( 1201 \ \mathrm{ml} \). Underlying misconception: Thinks a difference of one part in a ratio means the quantities will differ by one unit. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated misconceptions instead of addressing the student's actual confusion about how range changes under transformations, making no meaningful pedagogical contribution.

**Pass 2 (R6, persona-visible):** The teacher repeatedly provides irrelevant, copy-pasted examples about unrelated misconceptions instead of directly explaining how to calculate range or providing a worked example for the plant heights problem, which completely fails to instantiate the traditional teaching strategy.

---

## case_3 / constructive / baseline / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Can you identify any other properties of a rectangle that may help determine the angles in the figure?

**Turn 2 [student]**

> I think I remember that in a rectangle, opposite angles are equal and all the angles are right angles. So, if all the angles are right angles, does that mean they can't be acute? I'm still a bit confused about how that works with the diagonals though. Can you help me understand that part?

**Turn 3 [teacher]**

> Let's continue building on what you know about rectangles. You're correct that all angles in a rectangle are right angles. Now, think about how that property relates to the diagonals of a rectangle.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses generic prompting about rectangle properties without addressing the student's specific confusion about angles formed by diagonals at the center, and the student remains confused about the actual question by Turn 2, showing minimal progress toward understanding that central angles formed by diagonals differ from the rectangle's corner angles.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the constructive strategy by consistently scaffolding from the student's existing knowledge (rectangle properties), breaking the problem into manageable steps (first identifying properties, then connecting to diagonals), and adjusting support by acknowledging correct understanding while guiding toward the next conceptual step.

---

## case_3 / constructive / baseline / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Can you identify any other properties of rectangles that may help you determine the angles in the figure?

**Turn 2 [student]**

> Um, well I know that all the angles in a rectangle are 90 degrees, right? But I'm not sure how that helps with the angles in the middle where the diagonals cross... are those angles also 90 degrees?

**Turn 3 [teacher]**

> Let's continue building on what you know about rectangles. Can you think about how the property of all angles being 90 degrees in a rectangle can help you determine the angles where the diagonals cross?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher redirects to rectangle properties but doesn't address the student's specific misconception that all angles in the figure are equal, and by Turn 2 the student still incorrectly wonders if the diagonal angles are 90 degrees, showing minimal progress toward understanding that angles formed by diagonals in a non-square rectangle vary.

**Pass 2 (R6, persona-visible):** While the teacher attempts to scaffold by redirecting to known properties, they fail to break the problem into manageable steps or adjust support when the student explicitly signals confusion about how to connect rectangle angles to diagonal intersection angles.

---

## case_3 / constructive / hybrid / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> You're on the right track in considering the angles of the rectangle, but remember that rectangles have specific properties related to their diagonals. Let's focus on the intersecting diagonals of a rectangle and how they can help us determine the angles. Can you think about how the intersecting diagonals might affect the angles within the rectangle?

**Turn 2 [student]**

> I guess the diagonals might create some angles that aren't all equal, but I'm still not sure how that works. Like, if the diagonals cross, do they make the angles at the center different from the corners? Can you explain how that affects the angles?

**Turn 3 [teacher]**

> You're on the right track with considering how the intersecting diagonals of a rectangle can affect the angles. Remember, the intersecting diagonals of a rectangle bisect each other, creating congruent angles at the center. This property can help us determine the angles within the rectangle. Let's explore how the intersecting diagonals divide the rectangle and how that impacts the angles at the center and corners.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to address the student's specific misconception that all angles in the figure are equal, provides mathematically misleading information about congruent angles at the center (which contradicts the problem setup showing different colored angles), and the student shows no improvement, ending more confused than at the start.

**Pass 2 (R6, persona-visible):** While the teacher attempts to break down the problem and provide some scaffolding, they fail to adjust support based on the student's explicit request for explanation in Turn 2, instead repeating similar prompts rather than providing the more direct guidance the student clearly needs.

---

## case_3 / constructive / hybrid / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> It seems like you may not be familiar with the properties of a rectangle. Let's explore the intersecting diagonals of a rectangle and how they relate to the angles. Can you think about why all the angles in a rectangle are not necessarily equal?

**Turn 2 [student]**

> Oh right, yeah... I think you mean the corners of the rectangle are all 90 degrees, not the angles where the diagonals cross in the middle? So the red and yellow angles at the centre aren't the same as the corner angles? I'm still a bit confused about what angles the question is actually asking about though - is it asking about the angles made where the diagonals meet?

**Turn 3 [teacher]**

> You're on the right track! Remember, in a rectangle, the diagonals intersect at right angles, creating four congruent right triangles. Think about how the properties of these triangles relate to the angles at the center of the rectangle. Keep exploring how the intersecting diagonals impact the angles you're considering.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes a critical mathematical error in Turn 3 by incorrectly stating that rectangle diagonals intersect at right angles (they don't), which undermines the entire dialogue despite some attempt to engage with the student's confusion about which angles are being discussed.

**Pass 2 (R6, persona-visible):** While the teacher attempts to break down the problem and provide scaffolding, Turn 3 contains a significant mathematical error (stating that rectangle diagonals intersect at right angles, which is only true for squares), undermining the constructive approach and potentially confusing rather than supporting the student.

---

## case_3 / constructive / pure_ai / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> You're on the right track in recognizing that a rectangle has equal opposite angles, but it's important to remember that all four angles in a rectangle are right angles. Can you think about why that is the case in terms of the sides of a rectangle?

**Turn 2 [student]**

> Oh, I see! So all the angles in a rectangle are actually right angles, which means they’re 90 degrees. But I’m still a bit confused about why that is. Is it because of the way the sides are? Like, do the lengths of the sides matter for the angles?

**Turn 3 [teacher]**

> That's right! The reason why all angles in a rectangle are right angles is because a rectangle has opposite sides that are equal in length and parallel. This special property of rectangles helps create those 90-degree angles at each corner. Can you think of how having equal sides and parallel sides leads to right angles in a rectangle?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses rectangle properties generally but misses the student's actual confusion about the angles formed by diagonals (not corner angles), provides mathematically incomplete explanations about why rectangles have right angles, and the student shows no evidence of resolving their original misconception about the diagonal angles.

**Pass 2 (R6, persona-visible):** While the teacher breaks down concepts and asks guiding questions, they provide direct explanations rather than scaffolding from the student's actual understanding, and they don't address the student's original misconception about the diagonal angles being equal or acute.

---

## case_3 / constructive / pure_ai / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> You've made a good observation by assuming all angles are equal, but remember that a rectangle has specific properties. Can you think about what those properties might be and how they relate to the angles in a rectangle? Let's explore the characteristics of a rectangle together.

**Turn 2 [student]**

> Um, okay... so I know rectangles have four corners and they're all right angles, like 90 degrees. But I'm still confused about those angles in the middle where the diagonals cross - are those also 90 degrees, or can they be different?

**Turn 3 [teacher]**

> You're on the right track with recognizing that rectangles have four right angles. Remember that the diagonals of a rectangle bisect each other at the center, creating congruent triangles. Think about what that means for the angles where the diagonals intersect in relation to the properties of a rectangle.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher vaguely redirects to rectangle properties without addressing the specific misconception about central angles, provides mathematically misleading information (diagonals don't create congruent triangles in non-square rectangles, and central angles aren't necessarily related to rectangle properties in the way implied), and the student remains confused about whether central angles can vary.

**Pass 2 (R6, persona-visible):** While the teacher attempts to scaffold by acknowledging the student's thinking and prompting reflection on rectangle properties, they fail to break down the problem into manageable steps or adjust support when the student explicitly states confusion in Turn 2, instead providing another abstract hint rather than more concrete guidance.

---

## case_3 / constructive / pure_cbr_llm / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's focus on the properties of rectangles. Remember that rectangles have opposite sides that are equal in length and all angles are right angles. This can help us determine the correct angles in the figure you provided. Can you think about how these properties apply in this case?

**Turn 2 [student]**

> Okay, so if all the angles in a rectangle are right angles, then they can't be acute. I guess that means the angles in the figure can't be less than 90 degrees. But what about the angles where the diagonals cross? Are those angles also right angles?

**Turn 3 [teacher]**

> That's a good observation! Remember that rectangles have intersecting diagonals that bisect each other and create four right angles. This property can help us determine the angles where the diagonals cross. Can you think about how this information applies to the figure you provided?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming diagonals of rectangles create four right angles at their intersection, which is only true for squares), fails to address the student's specific misconception about angles at the center being equal, and the student remains confused about the central angles by Turn 2.

**Pass 2 (R6, persona-visible):** The teacher attempts to break down the problem and provide hints, but provides mathematically incorrect information (claiming rectangle diagonals create four right angles at their intersection, which is only true for squares) rather than scaffolding the student's own reasoning about the acute/obtuse angles formed.

---

## case_3 / constructive / pure_cbr_llm / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's focus on the properties of a rectangle. Remember, a rectangle has opposite sides that are equal in length and all angles are right angles. Can you think about how these properties relate to the angles highlighted in the figure you mentioned?

**Turn 2 [student]**

> Oh right, so all the angles in a rectangle are 90 degrees... but I'm still confused about those angles in the middle where the diagonals cross. Are you saying those angles at the centre can be acute, or are they also 90 degrees? I thought because it's a rectangle, everything inside it would have to follow the same rules.

**Turn 3 [teacher]**

> Let's focus on the intersecting diagonals properties of a rectangle. Can you think about how the intersecting diagonals relate to the angles at the centre of the rectangle?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically correct information but fails to address the student's specific confusion about angles at the diagonal intersection, and the student shows no progress, remaining confused about whether central angles are acute or 90 degrees by the end.

**Pass 2 (R6, persona-visible):** The teacher repeatedly asks vague, open-ended questions without breaking the problem into manageable steps or adjusting support when the student explicitly expresses confusion, failing to provide the scaffolding characteristic of constructive teaching.

---

## case_3 / constructive / pure_cbr_tpl / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that the angles in a rectangle aren't all equal, but I'm still confused about why they can't be acute. So, are you saying that the angles in a rectangle are always right angles? What does that mean for the angles at the center where the diagonals cross?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples from unrelated problems rather than addressing the student's specific misconception about angles in rectangles, and the student shows no improvement in understanding by the end of the dialogue.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical boilerplate text about similar cases without scaffolding, breaking down the problem into steps, or adjusting support based on the student's specific question about angles at the center where diagonals cross.

---

## case_3 / constructive / pure_cbr_tpl / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying not all the angles are the same? But I'm still confused about what happens when the diagonals cross in the middle - like, why would those angles at the centre never be acute? Can you explain what makes them different from the corners of the rectangle?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only generic, repetitive text about 'similar cases' without addressing the student's specific question about why angles at the center differ from corner angles, resulting in no visible progress in the student's understanding.

**Pass 2 (R6, persona-visible):** The teacher simply repeats identical boilerplate text about similar cases without scaffolding, breaking down the problem into steps, or adjusting support based on the student's specific confusion about diagonal angles, showing no implementation of constructive teaching strategy.

---

## case_3 / experiential / baseline / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think of a rectangle like a dinner plate - if you cut it in half diagonally, would the two halves have the same angles at the center? Imagine you were doing this with everyday objects you can see and touch — would your method still give the right answer?

**Turn 2 [student]**

> I think I get what you're saying about cutting it in half diagonally. So, if I imagine a dinner plate, the angles at the center would actually be the same, right? But then, how do I know what the angles are in a rectangle? I thought they were all the same, but now I'm not sure.

**Turn 3 [teacher]**

> Think of a rectangle like a window pane - when you look at the corners, are they all the same? Imagine you were doing this with everyday objects you can see and touch — would your method still give the right answer?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague analogies that confuse rather than clarify the distinction between rectangle corner angles and diagonal intersection angles, failing to address the student's specific misconception about angles formed by diagonals, and the student becomes more confused rather than improving.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete everyday analogies (dinner plate, window pane) to help the student visualize and understand the abstract geometric properties of rectangles, which is a strong implementation of the experiential strategy, though the analogies could be slightly more precise in addressing the specific diagonal angle question.

---

## case_3 / experiential / baseline / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Imagine you have a rectangular table in front of you. If you measure the angles at each corner, you'll notice that they are all equal. This is because a rectangle has four right angles, making it a special type of quadrilateral. So, when looking at the angles in a rectangle, you can always assume they are equal.

**Turn 2 [student]**

> Oh right, so the corners of the rectangle are all 90 degrees - I get that bit. But wait, I'm confused about the angles in the middle where the diagonals cross... are those the same as the corner angles too? Because those are the ones highlighted in red and yellow in the picture, not the corners.

**Turn 3 [teacher]**

> Imagine you have a rectangular window in front of you. The diagonal lines that cross in the middle create four right angles at each corner, just like the corners of the window. So, the angles in the middle where the diagonals cross are also 90 degrees, just like the corner angles. Would your method still give the right answer if you think about the window example?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming the angles where diagonals cross are 90 degrees, when they vary based on rectangle proportions), fails to address the student's actual confusion about central angles versus corner angles, and the student shows no improvement in understanding by the dialogue's end.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete real-world analogies (rectangular table, rectangular window) to ground the abstract geometric concepts, though the mathematical content in Turn 3 is incorrect (the diagonal intersection angles are not necessarily 90 degrees), which slightly undermines the pedagogical effectiveness.

---

## case_3 / experiential / hybrid / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Imagine you have a rectangular piece of paper. If you fold it along the diagonals, what do you notice about the angles where the folds meet? This can help us understand why the angles highlighted on the rectangle with different length sides can never be equal. Let's explore this together.

**Turn 2 [student]**

> I see what you mean about folding the paper, but I'm still a bit confused. So, are you saying that the angles where the folds meet are different because the sides of the rectangle are different lengths? But if all the angles in a rectangle are supposed to be right angles, how can they be anything else?

**Turn 3 [teacher]**

> Imagine you have a rectangular swimming pool with different lengths for the sides. When you fold a pool noodle diagonally along the longer side, what do you notice about the angles where the noodle meets the edges of the pool? This can help us understand why the angles on a rectangle with different length sides can never be equal, even though all the angles in a rectangle are right angles. Let's explore this together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague analogies that don't address the student's confusion between rectangle corner angles and central angles formed by diagonals, the student remains confused throughout, and the teacher's repeated unclear metaphors fail to scaffold appropriately or improve understanding.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, everyday analogies (folding paper, swimming pool with pool noodle) to help the student visualize and understand the abstract geometric concept of angles formed by diagonals in a rectangle.

---

## case_3 / experiential / hybrid / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Imagine you have a rectangle and you draw two diagonals that intersect in the middle. Just like how you need to remember that a rhombus has equal sides, in a rectangle, the intersecting diagonals will always create four right angles. So, in this case, all the angles in the figure will actually be right angles, not acute. Let's think about why that is and how it connects to the properties of a rectangle.

**Turn 2 [student]**

> Wait, so you're saying the diagonals in a rectangle always make right angles where they cross in the middle? But I thought that only happened in squares... does that mean every rectangle's diagonals do that, even the ones with different length sides?

**Turn 3 [teacher]**

> Imagine you have a rectangle with different length sides. Just like how you need to recall that a rhombus has equal sides, in a rectangle, the intersecting diagonals will always create four right angles. This property holds true for all rectangles, not just squares. Let's explore why this is the case and how it relates to the unique characteristics of rectangles.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (rectangle diagonals do not intersect at right angles), fails to engage with the student's actual reasoning about angle equality, and the student's Turn 2 response shows they are appropriately skeptical of the false claim being made.

**Pass 2 (R6, persona-visible):** The teacher makes no attempt to connect the abstract geometry to concrete real-world analogies or everyday situations, instead relying on abstract geometric properties and comparisons to other shapes like rhombuses.

---

## case_3 / experiential / pure_ai / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Imagine you have a rectangular piece of paper, and you fold it in half diagonally. Can you see how the angles formed at the intersection of the fold are always equal? This is because of a property of rectangles that all their angles are right angles, or 90 degrees. So, in a rectangle, opposite angles are always equal.

**Turn 2 [student]**

> Okay, I think I get that the angles in a rectangle are all right angles, so they can't be acute. But what if the rectangle has different lengths for the sides? Do the angles still stay the same? I'm a bit confused about how that works.

**Turn 3 [teacher]**

> Imagine you have a rectangular swimming pool with one long side and one short side. When you walk around the pool, you'll notice that the angles at the corners are still right angles, even though the sides are different lengths. This is because in a rectangle, the opposite sides are always parallel and the angles are always right angles, no matter the lengths of the sides.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher misidentifies the student's misconception (which is about angles formed by diagonals at the center, not corner angles of the rectangle), provides only declarative statements without requiring reasoning, and the student's final turn shows they remain confused about the actual problem involving diagonal intersections.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, everyday analogies (folding paper, walking around a swimming pool) to explain the abstract geometric properties of rectangles, which is a textbook example of the experiential strategy.

---

## case_3 / experiential / pure_ai / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Imagine the rectangle as a room in your house. The corners where the walls meet create right angles, just like the corners of a rectangle. This can help you remember that all angles in a rectangle are right angles, not acute angles.

**Turn 2 [student]**

> Oh right, so the corners of the rectangle are all 90 degrees... but I'm still confused about those angles in the middle where the diagonals cross. Are those the ones the question is asking about? Because those don't look like they'd be right angles.

**Turn 3 [teacher]**

> Imagine you're playing a game of tic-tac-toe on a rectangular board. The diagonals that cross in the center create a perfect X shape, with each angle measuring 90 degrees. This can help you see that the angles in the middle of a rectangle, where the diagonals intersect, are also right angles.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides incorrect mathematical content (claiming diagonal intersections form 90-degree angles when they don't in non-square rectangles), fails to engage with the student's actual confusion about the central angles versus corner angles, and simply tells answers rather than prompting reasoning, while the student's Turn 2 shows improved understanding of corner angles but receives misleading information about the central angles.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (room corners, tic-tac-toe), the explanations are mathematically incorrect (claiming diagonal intersections form 90-degree angles in all rectangles) and fail to genuinely ground the abstract concept in accurate experiential understanding.

---

## case_3 / experiential / pure_cbr_llm / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Think about a rectangle like a building with four walls. Each wall meets at a right angle, forming 90-degree angles at the corners. Just like how all the walls of a building are equal in length, all the sides of a rectangle are equal. So, the angles in a rectangle are always equal, not acute.

**Turn 2 [student]**

> Oh, I see! So all the angles in a rectangle are actually 90 degrees, which means they can't be acute at all. But what about the sides? I thought if the sides are different lengths, that might change the angles too. Can you explain that a bit more?

**Turn 3 [teacher]**

> Think of a rectangle like a soccer field - the length of the field is different from the width, but the angles at the corners where the sides meet are always 90 degrees. Just like how a soccer field can have different lengths and widths but still have 90-degree angles at the corners, a rectangle can have different side lengths but the angles will always be 90 degrees.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes a critical mathematical error by stating all sides of a rectangle are equal (confusing it with a square), provides only direct telling rather than eliciting reasoning, and while the student shows some progress in understanding corner angles are 90°, the fundamental misconception about the central angles formed by diagonals remains unaddressed.

**Pass 2 (R6, persona-visible):** The teacher consistently uses concrete, everyday analogies (building walls, soccer field) to explain the abstract geometric property that rectangles always have 90-degree angles regardless of side lengths, which is a textbook example of the experiential strategy.

---

## case_3 / experiential / pure_cbr_llm / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think of a rectangle like a soccer field - the intersecting diagonals are like the halfway line and the center circle. Just like how those lines divide the field into equal sections, the diagonals of a rectangle divide it into equal triangles. So, the angles at the center can never be equal in a rectangle.

**Turn 2 [student]**

> Oh right, so the diagonals don't make equal angles at the center... but I'm still a bit confused - does that mean the angles can never be acute, or just that they're not all the same? Like, couldn't some of them still be acute angles?

**Turn 3 [teacher]**

> Think of a rectangle like a swimming pool - the corners are like the shallow end and the deep end. Just as the corners of a rectangle are always right angles, meaning they can never be acute, the angles at the center can never be equal in a rectangle.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides confusing and mathematically incorrect statements (claiming angles at the center can never be equal and never be acute, when they actually are equal and can be acute or obtuse), uses unhelpful analogies, and the student remains confused, showing minimal improvement in understanding.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (soccer field, swimming pool), these analogies are superficial and don't actually ground the mathematical concepts in concrete experiences that clarify the geometric relationships—the soccer field analogy incorrectly explains the angle relationships, and the swimming pool analogy conflates corner angles with center angles without meaningful connection.

---

## case_3 / experiential / pure_cbr_tpl / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that the angles in a rectangle aren't all equal, but I'm still confused about why they can't be acute. I thought if the sides are different lengths, the angles could change too. Can you help me understand how the angles work in a rectangle?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant boilerplate text about unrelated problems rather than addressing the student's specific confusion about rectangle properties and diagonal angles, resulting in no progress on the misconception.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations whatsoever, instead repeatedly listing abstract similar cases and misconceptions without any experiential grounding.

---

## case_3 / experiential / pure_cbr_tpl / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying not all the angles at the centre are the same? But I thought because it's a rectangle, everything would be equal... Can you explain what actually happens when the diagonals cross in the middle?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical boilerplate text from unrelated problems without addressing the student's specific question about diagonal angles, providing no actual mathematical instruction or engagement with the misconception.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations, instead repeatedly listing abstract similar cases with mathematical topics and misconceptions, which is completely inconsistent with the experiential strategy.

---

## case_3 / rule_based / baseline / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> There is a specific property of rectangles that you need to consider. Step 1 is to identify the type of shape you are working with, and step 2 is to remember that opposite angles in a rectangle are equal.

**Turn 2 [student]**

> Oh, I see! So you’re saying that in a rectangle, the opposite angles are equal, but what about the angles at the corners? Are they all right angles? I thought they could be different.

**Turn 3 [teacher]**

> There is a specific property of rectangles that you need to consider. Step 1 is to identify the type of shape you are working with, and step 2 is to remember that in a rectangle, all angles at the corners are right angles.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides direct statements of properties rather than engaging with the student's confusion about rectangle properties, and the student's Turn 2 response reveals the core misconception remains unaddressed, with no evidence of improved understanding by dialogue end.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedural steps (Step 1, Step 2) and provides direct corrective feedback about rectangle properties, though the approach is somewhat formulaic and could include more worked examples to be a perfect instantiation.

---

## case_3 / rule_based / baseline / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> There is a specific rule for rectangles. Step 1 is to remember that opposite angles are equal in a rectangle. Step 2 is to apply this rule to find the correct angle measurements.

**Turn 2 [student]**

> Oh right, so opposite angles are equal... but I'm still a bit confused about which angles you mean? Like, are you talking about the angles at the corners of the rectangle itself, or the angles where the diagonals cross in the middle? Because I was looking at those angles in the middle where the red and yellow ones are highlighted.

**Turn 3 [teacher]**

> There is a specific rule for rectangles. Step 1 is to remember that opposite angles are equal in a rectangle. Step 2 is to apply this rule to the angles at the corners of the rectangle itself, not the angles where the diagonals cross in the middle.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly states an irrelevant rule about rectangle corner angles when the student is asking about diagonal intersection angles, fails to address the actual misconception about angles formed by diagonals, provides no reasoning opportunities, and the student shows no improvement in understanding by the end.

**Pass 2 (R6, persona-visible):** The teacher states explicit procedural steps and attempts to give direct feedback, but the rule stated (opposite angles are equal in a rectangle) is misleading in this context since it applies to corner angles which are all 90°, and the teacher fails to directly address the actual procedural error regarding the angles formed by diagonals at the center.

---

## case_3 / rule_based / hybrid / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> The correct answer is "never" because the intersecting diagonals in a rectangle always form right angles. Remember, in a rectangle, opposite angles are equal and the diagonals bisect each other at right angles. So, it's important to recognize these properties when determining the angles in a rectangle.

**Turn 2 [student]**

> Oh, I see! So the angles at the intersection of the diagonals in a rectangle are always right angles, which means they can't be acute. But does that mean all the angles in a rectangle are right angles, or just the ones where the diagonals cross? I'm still a bit confused about that.

**Turn 3 [teacher]**

> In a rectangle, only the angles where the diagonals intersect are guaranteed to be right angles. Remember, opposite angles in a rectangle are equal, but not all angles in a rectangle are necessarily right angles. It's important to differentiate between the properties of a rectangle and those specific to the intersecting diagonals.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (diagonals of rectangles do NOT bisect at right angles except in squares, and all corner angles of rectangles ARE right angles), directly tells answers without eliciting reasoning, and the student's final turn shows increased confusion rather than improvement.

**Pass 2 (R6, persona-visible):** While the teacher states some explicit rules about rectangles, the rules given are mathematically incorrect (diagonals of rectangles do not bisect at right angles, and all corner angles of rectangles are right angles), which undermines the rule-based approach and provides faulty procedural knowledge.

---

## case_3 / rule_based / hybrid / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> The correct property of a rectangle is that the intersecting diagonals are always equal in length and bisect each other. Since the angles highlighted on the rectangle with different length sides can never be equal, the assumption that all angles in the figure are equal is incorrect. Remember, in a rectangle, opposite angles are always equal.

**Turn 2 [student]**

> Oh right, so the diagonals bisect each other... but I'm still a bit confused about why the angles can't be acute then? Like, if the diagonals cut each other in half at the centre, couldn't those angles at the middle still be acute angles sometimes?

**Turn 3 [teacher]**

> The key property of a rectangle is that the diagonals bisect each other, not necessarily cut each other in half at the center. This means that the angles at the center of the rectangle will always be right angles, not acute angles. Remember, a rectangle has four right angles by definition.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming angles at the center where diagonals meet are always right angles, and confusing 'bisect each other' with something else), simply tells answers rather than prompting reasoning, and the student remains confused throughout with no visible improvement in understanding.

**Pass 2 (R6, persona-visible):** While the teacher states some explicit rules about rectangles, they provide incorrect information (claiming diagonals create right angles at the center, confusing rectangle corner angles with diagonal intersection angles), lack worked steps or procedural demonstrations, and don't give clear corrective feedback on the actual procedural error in the student's reasoning.

---

## case_3 / rule_based / pure_ai / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> The properties of a rectangle include having four right angles. In a rectangle, all four angles are equal to 90 degrees. In this case, assuming all angles are equal is incorrect because only rectangles have all angles equal to 90 degrees. Remember, in a rectangle, opposite sides are equal in length and parallel.

**Turn 2 [student]**

> Oh, I see! So all the angles in a rectangle are actually 90 degrees, which means they can’t be acute at all. But what if the rectangle has different lengths for the sides? Does that change anything about the angles?

**Turn 3 [teacher]**

> Yes, that's correct! Even if the sides of a rectangle have different lengths, all four angles will still be 90 degrees. The property of having four right angles in a rectangle does not change based on the lengths of the sides.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher directly tells the student facts about rectangles without engaging with the actual misconception about the central angles formed by diagonals (which are not 90 degrees), and the student's improved understanding in Turn 2 is about rectangle corner angles, not the diagonal angles that were the original question.

**Pass 2 (R6, persona-visible):** While the teacher states some explicit facts about rectangles, they do not provide worked steps showing how to analyze the highlighted angles at the center where diagonals intersect, nor do they directly address the procedural error in the student's reasoning about those specific angles.

---

## case_3 / rule_based / pure_ai / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> The properties of a rectangle include having all angles equal to 90 degrees. In a rectangle, opposite sides are equal in length and parallel. Remember to consider these properties when identifying angles in a rectangle.

**Turn 2 [student]**

> Oh right, so all the corner angles of the rectangle are 90 degrees - I get that bit now. But I'm still confused about those highlighted angles in the middle where the diagonals cross... are those the ones we're looking at, or are we talking about the corners? Because those angles in the middle don't look like they'd all be the same size to me.

**Turn 3 [teacher]**

> In a rectangle, the diagonals intersect at right angles, forming four congruent right triangles. The angles at the intersection of the diagonals are all right angles, each measuring 90 degrees. Remember, in a rectangle, all angles are 90 degrees.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (diagonals of a rectangle do not intersect at right angles unless it's a square), simply tells answers without engaging the student's reasoning about the central angles, and the student shows no improvement in understanding by the end.

**Pass 2 (R6, persona-visible):** While the teacher states explicit rules about rectangle properties, they provide incorrect information (diagonals of rectangles do not generally intersect at right angles) and fail to give direct corrective feedback on the student's specific procedural error in analyzing the central angles formed by the diagonals.

---

## case_3 / rule_based / pure_cbr_llm / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> The correct answer for the angles highlighted on a rectangle with different length sides is "never." This is because the intersecting diagonals in a rectangle create right angles. Make sure to recall and use the intersecting diagonals properties of a rectangle to determine the correct answer.

**Turn 2 [student]**

> Oh, I see! So the angles where the diagonals cross in a rectangle are right angles, which means they can't be acute. But what about the other angles in the rectangle? Are they all right angles too?

**Turn 3 [teacher]**

> The other angles in a rectangle are not necessarily right angles. The only angles that are guaranteed to be right angles are the ones where the diagonals intersect. Remember to identify a rectangle by recalling and using the properties of its intersecting diagonals.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming diagonals of rectangles create right angles and that corner angles aren't necessarily right angles), simply tells answers rather than prompting reasoning, and the student's final question suggests confusion about basic rectangle properties remains unresolved.

**Pass 2 (R6, persona-visible):** The teacher states some rules about diagonals but provides incorrect information (claiming rectangle corners aren't right angles, and that diagonals create right angles at intersection), lacks worked examples or step-by-step procedures, and doesn't directly correct the student's procedural reasoning.

---

## case_3 / rule_based / pure_cbr_llm / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> The properties of a rectangle include having opposite sides that are equal in length and all angles being 90 degrees. Since all angles in a rectangle are equal, it is not safe to assume that all angles in the figure are equal. Remember, a rectangle is a special type of parallelogram with four right angles.

**Turn 2 [student]**

> Oh right, so the rectangle itself has all 90 degree angles, but the angles at the centre where the diagonals cross are different? So those angles in the middle aren't the same as the corners of the rectangle?

**Turn 3 [teacher]**

> Yes, that's correct. The intersecting diagonals of a rectangle create four congruent right triangles, so the angles at the center where the diagonals cross are all 90 degrees. This is a unique property of rectangles.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides direct statements rather than engaging the student's reasoning, and critically, gives mathematically incorrect information in Turn 3 (the angles at the center where diagonals cross are NOT all 90 degrees in a non-square rectangle), which the student appears to accept without developing genuine understanding.

**Pass 2 (R6, persona-visible):** The teacher explains properties and concepts but does not state explicit procedures, show worked steps, or give direct corrective feedback on procedural errors as required by the rule-based strategy.

---

## case_3 / rule_based / pure_cbr_tpl / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I still don’t really get why the angles can never be acute. I thought since the rectangle has equal angles, they could all be the same size, like in a square. Can you explain why that’s not right?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides identical, generic lists of unrelated example problems twice instead of addressing the student's specific confusion about angles formed by diagonals in rectangles, resulting in no progress toward correcting the misconception.

**Pass 2 (R6, persona-visible):** The teacher never states explicit procedures, shows worked steps, or gives direct corrective feedback; instead, they only present vague references to similar cases without any rule-based instruction.

---

## case_3 / rule_based / pure_cbr_tpl / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying not all the angles are the same? But I'm still confused about what happens when the diagonals cross in the middle - like, how do I know whether those angles at the centre are acute or not if I don't know their exact sizes?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats identical generic case examples twice without addressing the student's specific confusion about diagonal angles, provides no mathematical explanation, and the student shows no improvement in understanding.

**Pass 2 (R6, persona-visible):** The teacher does not state any explicit procedures, show worked steps, or give direct corrective feedback; instead, they repeatedly present generic similar cases without addressing the student's specific question about determining angle types at the intersection of diagonals.

---

## case_3 / socratic / baseline / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> What if we consider the properties of a rectangle where all angles are not necessarily equal? How would that change your approach to determining the angles in the figure?

**Turn 2 [student]**

> I guess if not all angles are equal, then maybe the angles could be different? But I thought rectangles always had right angles, so how can they be anything else? I'm still a bit confused about what that means for the angles in the rectangle.

**Turn 3 [teacher]**

> Let me ask you this: what would happen if you considered what makes a rectangle unique in terms of its angles? How might that understanding help you determine the angles in this figure?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher fails to address the student's core confusion about rectangle properties (the student actually knows rectangles have right angles but is confused by the teacher's misleading question), provides vague questions that confuse rather than clarify, and the student shows no improvement—ending more confused than they started.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward understanding rectangle properties without directly stating the answer, though they don't employ counterexamples to challenge the student's flawed reasoning about angles being unequal.

---

## case_3 / socratic / baseline / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> If all the angles in the figure were equal, what would that mean about the properties of a rectangle? Let me ask you this: what would happen if you tried your method on a simpler version of the problem?

**Turn 2 [student]**

> Hmm, if all the angles were equal, then... I guess that would mean it's a square? But you said it's a rectangle with different length sides, so that doesn't work. I'm confused though - what do you mean by trying it on a simpler version? Do you mean like just looking at the angles without the diagonals?

**Turn 3 [teacher]**

> If all the angles were equal, it would indeed be a square, but since we have a rectangle with different length sides, that doesn't apply here. Let me ask you this: what would happen if you focused on the properties of a rectangle without considering the diagonals?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher vaguely gestures at the misconception but never directly addresses the student's faulty assumption about angles being equal, the questions are unclear and don't scaffold effectively, and the student ends more confused than they started, showing no improvement in understanding.

**Pass 2 (R6, persona-visible):** While the teacher does ask probing questions, they directly tell the student the answer in Turn 3 ('If all the angles were equal, it would indeed be a square, but since we have a rectangle with different length sides, that doesn't apply here'), which violates the core Socratic principle of never giving answers directly.

---

## case_3 / socratic / hybrid / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Can you explain why you assumed all angles in the figure are equal? How does the definition of a rectangle relate to the angles within it?

**Turn 2 [student]**

> I thought all the angles were equal because I remembered something about shapes having equal angles, but I guess I didn’t really think about what a rectangle actually is. I know rectangles have right angles, but I’m confused about how that works with the angles in the middle where the diagonals cross. Can you help me understand that better?

**Turn 3 [teacher]**

> Can you recall any properties of rectangles that might help you understand how the angles in the middle where the diagonals cross behave? How do the intersecting diagonals of a rectangle relate to the angles within it?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks generic questions about rectangle properties without addressing the specific misconception about angles formed by diagonals, and the student remains confused by Turn 2 without making progress toward understanding why diagonal intersection angles differ from corner angles.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without ever directly providing answers, which is a textbook example of the Socratic method.

---

## case_3 / socratic / hybrid / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Why do you think all the angles in a rectangle are equal? Can you think of any specific properties of rectangles that might help us determine the angles in this particular case?

**Turn 2 [student]**

> Um, I think I was confusing it with a square maybe? I know rectangles have four right angles at the corners, but I'm not really sure how that helps with the angles in the middle where the diagonals cross... Does that mean the angles at the centre aren't right angles then?

**Turn 3 [teacher]**

> Why do you think the fact that rectangles have four right angles at the corners is important when considering the angles in the middle where the diagonals cross? Can you think of any other properties of rectangles that might help us determine the angles at the center of the rectangle?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks open questions requiring reasoning but fails to address the student's confusion about diagonal angles, leaving them still uncertain about the problem after recognizing their initial error about confusing rectangles with squares.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never provides direct answers, and helps the student recognize their flawed reasoning about angle equality without explicitly correcting them.

---

## case_3 / socratic / pure_ai / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Can you explain why you assumed all the angles in the figure are equal? How does that assumption relate to the properties of a rectangle?

**Turn 2 [student]**

> I thought all the angles in a rectangle are the same because I remember something about equal angles in shapes. But now I'm not sure if that's right for rectangles. Can rectangles have different angles, or are they all supposed to be the same?

**Turn 3 [teacher]**

> Can you think of any other properties of a rectangle that might help you determine if the angles are all the same or not? How do the sides of a rectangle relate to its angles?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks generic questions about rectangle properties without addressing the specific confusion between corner angles and central angles formed by diagonals, and the student becomes more confused rather than progressing toward understanding.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without ever providing direct answers, which is a textbook example of the Socratic method.

---

## case_3 / socratic / pure_ai / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Why do you assume that all angles in a rectangle are equal? Can you think of any properties of rectangles that might help you determine the angles in this figure?

**Turn 2 [student]**

> Um, I think I was wrong about that... I know rectangles have four right angles at the corners, so those are all 90 degrees. But I'm not really sure how that helps with the angles in the middle where the diagonals cross? Are those angles also 90 degrees, or can they be different?

**Turn 3 [teacher]**

> That's a great observation about the right angles at the corners of a rectangle. Now, think about how the diagonals of a rectangle bisect each other at right angles. How does that property help us determine the angles where the diagonals cross?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes a critical mathematical error in Turn 3 by stating that rectangle diagonals bisect each other at right angles (they don't—only square diagonals do), which undermines the entire dialogue despite some attempt to engage the student's reasoning about angle properties.

**Pass 2 (R6, persona-visible):** While the teacher uses questions throughout, they directly state an incorrect property in Turn 3 (that rectangle diagonals bisect each other at right angles, which is false—only in squares), violating the Socratic principle of guiding students to discover information rather than providing it.

---

## case_3 / socratic / pure_cbr_llm / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Why do you think all the angles in a rectangle are equal? Can you think of a property of rectangles that might help you determine the angles in this particular case?

**Turn 2 [student]**

> I thought all the angles were equal because it looks like a square, but I guess rectangles can have different lengths for the sides. So, are the angles in a rectangle always right angles? I'm not really sure how that works with the diagonals.

**Turn 3 [teacher]**

> Why do you think the angles in a rectangle should always be right angles? Can you recall any properties of rectangles that might help you determine the angles formed by the diagonals?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly asks open questions about rectangle properties but doesn't address the student's actual confusion about angles formed by diagonals, and the student shows no progress in understanding what angles the question is actually asking about (angles at the center formed by diagonals, not the rectangle's corner angles).

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without ever directly providing answers, which is a textbook example of the Socratic method.

---

## case_3 / socratic / pure_cbr_llm / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Can you explain why you assumed all the angles in the figure are equal, even though the sides have different lengths?

**Turn 2 [student]**

> Um, I think I just thought that because it's a rectangle, all the angles would be the same... but now you're saying that the different side lengths matter? I'm confused though - doesn't a rectangle always have the same angles at the corners anyway?

**Turn 3 [teacher]**

> Can you recall any specific properties of rectangles that might help clarify the relationship between the angles and side lengths in this figure?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks generic questions about properties without addressing the specific confusion between corner angles and central angles formed by diagonals, and the student shows no progress, remaining confused about the same issue across both turns.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without giving direct answers, asking the student to explain their reasoning and recall relevant properties rather than correcting misconceptions directly.

---

## case_3 / socratic / pure_cbr_tpl / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I see what you're saying, but I'm still a bit confused. So, are you saying that the angles in a rectangle can never be acute because they have to be right angles? But then why did I think they could be equal? Can you help me understand that part better?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher merely repeats generic case examples twice without addressing the student's specific confusion about angles at the center where diagonals intersect versus corner angles of the rectangle, providing no actual teaching or engagement with the misconception.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions or use counterexamples; instead, they simply present lists of similar cases with correct answers and misconceptions, which directly contradicts the Socratic method of guiding through questioning.

---

## case_3 / socratic / pure_cbr_tpl / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying not all the angles are the same? But I'm still confused about what happens when the diagonals cross in the middle - like, how do I know whether those angles at the centre are acute or not if I don't know how long the sides are?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides identical, generic case lists twice without addressing the student's specific confusion about diagonal angles, offers no mathematical reasoning or scaffolding, and the student shows no improvement—remaining confused about the same concept by the end.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions or use counterexamples; instead, they simply present lists of similar cases with misconceptions and correct answers, which directly contradicts the Socratic method of guiding through questioning.

---

## case_3 / traditional / baseline / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let me explain the properties of a rectangle, which include having four right angles and opposite sides that are equal in length. Then, we will work through an example together to solidify your understanding. Can you tell me what you now understand about the angles in a rectangle?

**Turn 2 [student]**

> I think I get that rectangles have four right angles, so they can't be acute. But I'm still a bit confused about why I thought all the angles would be equal. Does that mean the angles at the corners are always the same in every rectangle?

**Turn 3 [teacher]**

> Let me clarify that in a rectangle, the opposite angles are equal, but the angles at the corners are not always the same. Let's work through an example together to see this in action. Can you think of any examples of rectangles you have seen before?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher makes a critical mathematical error in Turn 3 by stating that corner angles in rectangles are not always the same (they are always 90°), fails to address the actual misconception about angles formed by diagonals at the center, and the student shows no improvement in understanding the original problem about diagonal angles.

**Pass 2 (R6, persona-visible):** While the teacher attempts to explain concepts directly, they fail to provide worked examples as promised, make a significant mathematical error (claiming opposite angles in a rectangle are not always equal when all corner angles are always 90°), and rely heavily on questioning rather than clear direct explanation characteristic of traditional teaching.

---

## case_3 / traditional / baseline / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let me explain the properties of a rectangle to you. A rectangle has four right angles, meaning all angles are 90 degrees. Let's work through an example together to solidify this concept.

**Turn 2 [student]**

> Oh right, so the four corners of the rectangle are all 90 degrees - I get that bit now. But I'm still confused about those angles in the middle where the diagonals cross... are those 90 degrees too, or can they be different?

**Turn 3 [teacher]**

> Let me clarify that the angles in the middle where the diagonals cross in a rectangle are also 90 degrees. So, all angles in a rectangle are right angles. Now, let's work through an example to reinforce this understanding.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically incorrect information (claiming diagonals always intersect at 90 degrees in rectangles), uses direct telling rather than eliciting reasoning, and the student's final question suggests the misconception remains unaddressed.

**Pass 2 (R6, persona-visible):** While the teacher attempts direct explanation, they provide incorrect mathematical information (claiming all angles where diagonals cross are 90 degrees in any rectangle, which is only true for squares) and never actually works through the promised example, failing key aspects of the traditional strategy.

---

## case_3 / traditional / hybrid / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's revisit the properties of a rectangle. Remember, a rectangle has four right angles, opposite sides are equal in length, and the diagonals are equal in length and bisect each other. Can you identify these properties in the figure you mentioned?

**Turn 2 [student]**

> Okay, I see that a rectangle has four right angles, so I guess the angles in the middle should be 90 degrees. But I thought they could be acute because of the different lengths of the sides. Does that mean all the angles are always right angles, no matter what?

**Turn 3 [teacher]**

> You are on the right track in recognizing the right angles in a rectangle. Remember that all four angles in a rectangle are always right angles, regardless of the lengths of the sides. This property is essential in identifying rectangles and differentiating them from other quadrilaterals. Can you think of a way to apply this to the angles in the figure you mentioned?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly states rectangle properties but doesn't address the student's actual confusion about the central angles formed by diagonals (which are NOT right angles), and the student's final turn shows they still don't understand which angles in the figure can be acute.

**Pass 2 (R6, persona-visible):** The teacher uses questioning and prompting to guide discovery rather than directly explaining with worked examples, which is inconsistent with the traditional strategy of explicit instruction.

---

## case_3 / traditional / hybrid / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's review the properties of a rectangle together. In a rectangle, all angles are right angles, meaning they measure 90 degrees. This applies to all rectangles, regardless of the lengths of their sides. Can you identify any right angles in the figure you mentioned?

**Turn 2 [student]**

> Oh right, yeah, I see the four corners of the rectangle are all 90 degrees. But um... the angles you're asking about - the red and yellow ones - those are in the middle where the diagonals cross, not at the corners, so are those different? Like, do those have to follow the same rules as the corner angles?

**Turn 3 [teacher]**

> Let's focus on the properties of the intersecting diagonals in a rectangle. Remember, in a rectangle, the diagonals are always congruent and bisect each other. This means that the angles formed at the intersection of the diagonals are also equal. Can you think about how this relates to the red and yellow angles you mentioned?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses rectangle properties generically rather than engaging with the student's specific misconception about central angles formed by diagonals, and while the student shows some progress in distinguishing corner from central angles, the dialogue doesn't resolve whether they understand these central angles' properties or answer the original question.

**Pass 2 (R6, persona-visible):** The teacher uses questioning to guide discovery rather than directly explaining with worked examples, which is inconsistent with the traditional strategy of explicit instruction and demonstration.

---

## case_3 / traditional / pure_ai / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's clarify the properties of a rectangle. In a rectangle, opposite sides are equal in length and all angles are right angles. So, the angles at the centre of a rectangle are always 90 degrees. Can you identify the properties of the rectangle in the given figure now?

**Turn 2 [student]**

> Okay, I see that all the angles in a rectangle are right angles, so they are 90 degrees. But I'm still confused about the angles at the centre. If they’re all 90 degrees, does that mean they can’t be acute at all?

**Turn 3 [teacher]**

> Let's clarify further. In a rectangle, the angles at the centre are formed by the diagonals, which bisect each other and create four right angles. So, the angles at the centre are always 90 degrees, just like the other angles in a rectangle. Can you now see why the angles at the centre cannot be acute?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher incorrectly claims the central angles formed by diagonals are 90 degrees (they're not in a non-square rectangle), repeatedly tells rather than elicits reasoning, and the student shows no genuine understanding improvement, only confused agreement with false information.

**Pass 2 (R6, persona-visible):** The teacher provides incorrect mathematical explanations (falsely claiming angles at the center formed by diagonals are always 90 degrees in rectangles) rather than using accurate worked examples, which fundamentally violates the traditional strategy's emphasis on clear, correct direct instruction.

---

## case_3 / traditional / pure_ai / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> First, let's clarify the properties of a rectangle. A rectangle has four right angles, meaning all interior angles are 90 degrees. Since you assumed all angles in the figure are equal, that's why you thought they were acute. Can you identify the right angles in the rectangle you mentioned?

**Turn 2 [student]**

> Oh right, so the four corners of the rectangle are all 90 degrees - I can see that now. But I'm still confused about the angles in the middle where the diagonals cross... are those the ones we're looking at? Because those don't look like they're 90 degrees to me.

**Turn 3 [teacher]**

> Yes, you're correct that the four corners of the rectangle have right angles. However, the angles in the middle where the diagonals cross are not necessarily 90 degrees. Can you think about why that might be the case based on what we know about rectangles?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies rectangle properties but doesn't effectively address why the student thought all angles were equal, provides minimal scaffolding for reasoning about diagonal angles, and the student shows some progress in identifying corner angles but remains confused about the central angles without resolution.

**Pass 2 (R6, persona-visible):** The teacher asks probing questions rather than directly explaining with worked examples, which is more consistent with inquiry-based or Socratic methods than the traditional strategy of direct explanation.

---

## case_3 / traditional / pure_cbr_llm / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's recall the intersecting diagonals properties of a rectangle. The angles formed at the intersection are always right angles, not acute. Can you explain why this is the case?

**Turn 2 [student]**

> I think I get that the angles where the diagonals cross are right angles, but I'm still confused about why they can't be acute. I thought since the rectangle has different length sides, maybe that would change the angles too. Can you explain more about how the sides affect the angles?

**Turn 3 [teacher]**

> Let's focus on why the angles where the diagonals cross are always right angles. Remember, the properties of a rectangle guarantee that these angles will be right angles, regardless of the lengths of the sides. Let's walk through the reasoning together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher incorrectly claims that diagonals of a rectangle intersect at right angles (they don't - they bisect each other but form various angles depending on side lengths), fails to address the student's actual misconception about angle equality, and the student remains confused by the end showing no improvement in understanding.

**Pass 2 (R6, persona-visible):** The teacher repeatedly asks the student to explain or reason through the problem rather than directly explaining with worked examples, which contradicts the core traditional strategy of direct explanation.

---

## case_3 / traditional / pure_cbr_llm / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's recall and use the intersecting diagonals properties of a rectangle to determine the correct answer for the angles in this figure. Can you explain why you chose acute angles and how the properties of a rectangle come into play here?

**Turn 2 [student]**

> Um, well I thought all the angles would be the same because it's a rectangle, but I'm not really sure how the diagonals change things? Like, does the fact that the diagonals cross in the middle mean those angles at the centre are different from the corner angles of the rectangle?

**Turn 3 [teacher]**

> Let's focus on the intersecting diagonals properties of a rectangle. Remember that the diagonals of a rectangle are equal in length and bisect each other, creating four right angles at the intersection point. Can you see how this information helps us determine the correct angles in the figure?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher incorrectly states that diagonals of a rectangle create four right angles at their intersection (they don't unless it's a square), fails to address the student's actual misconception about angle equality, and the student shows no improvement in understanding by the end.

**Pass 2 (R6, persona-visible):** The teacher uses questioning and prompting rather than direct explanation with worked examples, which is inconsistent with the traditional strategy of explicitly explaining concepts and showing how to solve the problem step-by-step.

---

## case_3 / traditional / pure_cbr_tpl / leg_a

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I get that the angles in a rectangle aren't all equal, but I'm still confused about what types of angles they can be. So, if the angles aren't acute, what are they supposed to be? Do they have to be right angles or something?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only generic, repetitive case examples without addressing the student's specific confusion about rectangle properties or the angles formed by diagonals, and the student shows no improvement in understanding across the dialogue.

**Pass 2 (R6, persona-visible):** The teacher does not explain anything directly, provide worked examples, or check understanding; instead, they repeatedly copy-paste irrelevant case examples without addressing the student's specific question about angle types.

---

## case_3 / traditional / pure_cbr_tpl / leg_b

**Misconception:** Does not know the properties of a rectangle

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The angles highlighted on this rectangle with different length sides can never be... ![A rectangle with the diagonals drawn in. The angle on the right hand side at the centre is highlighted in red and the angle at the bottom at the centre is highlighted in yellow.](). I got acute because I assumed all the angles in the figure are equal.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying not all the angles are the same? But I'm still confused about what happens when the diagonals cross in the middle - like, how do I know whether those angles at the centre are acute or not if I don't know their exact sizes?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Recall and use the intersecting diagonals properties of a rectangle. Correct answer (C): never. Student selected: always. Underlying misconception: Does not know the properties of a rectangle. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Identify a rectangle when presented non-conventionally. Correct answer (A): Parallelogram. Student selected: Rhombus. Underlying misconception: Does not remember that a rhombus has equal sides. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Calculate the area of a rectangle where the dimensions are given in the same units. Correct answer (B): \( 40 \mathrm{~cm}^{2} \). Student selected: \( 13 \mathrm{~cm}^{2} \). Underlying misconception: Adds when calculating area instead of multiplying. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides identical, generic lists of unrelated example problems in both turns without addressing the student's specific confusion about diagonal angles in rectangles, resulting in no progress on the misconception.

**Pass 2 (R6, persona-visible):** The teacher does not explain anything directly, provide worked examples, or check understanding; instead, they repeatedly copy-paste irrelevant similar cases that don't address the student's specific question about angles at intersecting diagonals.

---

## case_4 / constructive / baseline / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Think about how the equation relates to the values in the table.

**Turn 2 [student]**

> Okay, so the equation \( f = 3r^{2} + 3 \) is used to find the values for \( f \) based on \( r \). I think I see that I should plug in the values of \( r \) into the equation to find \( f \) instead of just adding. But I'm still not sure how to do that for \( r = 3 \). Can you help me with that?

**Turn 3 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Think about how you can use the equation to find the value of \( f \) when \( r = 3 \).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never addresses the student's linear addition misconception, uses identical generic prompts twice, and fails to provide appropriate scaffolding when the student explicitly asks for help with substitution, though the student does show some self-correction by Turn 2.

**Pass 2 (R6, persona-visible):** While the teacher attempts to scaffold by redirecting to known information, they repeat the same vague prompt twice without breaking down the problem into manageable steps or adjusting support when the student explicitly asks for help with the specific calculation.

---

## case_4 / constructive / baseline / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Think about how the equation can help you find the missing value in the table.

**Turn 2 [student]**

> Oh right, so I should use the equation f = 3r² + 3 to work it out? So like, when r = 3, I plug that number into the equation instead of just adding on the difference? Let me try... 3 times 3 squared plus 3... is that 30?

**Turn 3 [teacher]**

> Let's go back to the part you do know, and build the next step from there. Think about how the equation f = 3r² + 3 can help you find the missing value in the table.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never acknowledges the student's linear-addition misconception and provides identical generic prompts even after the student self-corrects and attempts the right method, though the student does show improvement by recognizing they should use the equation rather than add differences.

**Pass 2 (R6, persona-visible):** While the teacher attempts to scaffold by redirecting to the equation, they simply repeat the exact same prompt in Turn 3 instead of adjusting support based on the student's progress in Turn 2, failing to break down the problem into manageable steps or respond to the student's specific computational attempt.

---

## case_4 / constructive / hybrid / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you may be treating the table as if the relationship between f and r is linear. However, remember that the equation provided is a quadratic equation. Let's explore how we can use the equation \( f=3r^{2}+3 \) to find the missing value in the table. Let's start by plugging in the given values of \( r \) and \( f \) to see if we can identify a pattern.

**Turn 2 [student]**

> Okay, I see what you mean about it being a quadratic equation. So, if I plug in \( r = 3 \) into the equation \( f = 3r^{2} + 3 \), I guess I should calculate it instead of just adding like I did before. But I'm not really sure how to do that. Can you help me figure out what \( f \) would be when \( r = 3 \)?

**Turn 3 [teacher]**

> You're on the right track by recognizing the need to calculate instead of just adding. To find the value of \( f \) when \( r = 3 \), substitute \( r = 3 \) into the equation \( f = 3r^{2} + 3 \) and then perform the necessary operations. Think about how to square \( 3 \) and then multiply by \( 3 \) before adding \( 3 \). Can you try that out and see what you get?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly identifies the linear-thinking misconception and redirects to the quadratic nature, but ultimately provides step-by-step instructions rather than eliciting deeper reasoning, and while the student recognizes they should substitute rather than add, they still request direct help rather than demonstrating independent understanding.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the constructive strategy by breaking down the problem into manageable steps (identifying the equation type, then substituting, then performing operations in sequence), scaffolding within the student's ZPD by providing just enough guidance without solving the problem, and adjusting support based on the student's demonstrated understanding of their error.

---

## case_4 / constructive / hybrid / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you may be treating the values in the table as linear, but in this case, we need to use the equation \( f=3r^{2}+3 \) to find the missing value. Think about how the equation relates to the values given in the table to determine the correct answer for the starred value.

**Turn 2 [student]**

> Oh right, so I need to use the equation instead of just adding on the difference? So like, if r = 3, I'd put that into the equation f = 3r² + 3... so that would be 3 times 3 squared plus 3? I'm still a bit confused about what 3 squared means though - is that 3 times 2?

**Turn 3 [teacher]**

> That's a great start! Remember, \(r^2\) means \(r\) multiplied by itself, so in this case, \(3^2\) would be \(3 \times 3\). Try substituting \(r = 3\) into the equation and see if you can calculate the value for \(f\). You're on the right track by using the equation to find the missing value in the table!

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly identifies and addresses the linear addition misconception, provides mathematically accurate guidance, and the student demonstrates clear progress from using linear differences to attempting to substitute into the quadratic equation, though a new confusion about squaring emerges.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the constructive strategy by breaking down the problem into manageable steps (first correcting the linear assumption, then addressing the specific misconception about squaring), scaffolding within the student's ZPD by providing just enough support without solving the problem, and adjusting the level of support based on the student's revealed confusion about exponents.

---

## case_4 / constructive / pure_ai / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> You're on the right track with using the equation, but remember that the relationship may not be linear. Try plugging in the values of \( r \) and see if the resulting \( f \) values match the table. Keep going with this approach to find the value of the star.

**Turn 2 [student]**

> Okay, I see what you mean about plugging in the values. So, if I try \( r = 1 \), I get \( f = 3(1^2) + 3 = 6 \), which matches. For \( r = 2 \), I get \( f = 3(2^2) + 3 = 15 \), and that matches too. But when I try \( r = 3 \), I get \( f = 3(3^2) + 3 = 30 \). Is that the right way to do it? So, the value for the star should be \( 30 \), right?

**Turn 3 [teacher]**

> You're doing a great job plugging in the values and noticing the pattern. Remember to check your calculations carefully for any mistakes. Take a closer look at how you calculated \( f \) for \( r = 3 \) and see if you can identify where you might have gone off track.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 5 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never directly addresses the linear addition misconception, but the student independently abandons it in Turn 2 by correctly using the quadratic formula and arrives at the correct answer (30), demonstrating clear improvement despite weak scaffolding.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the constructive strategy by scaffolding the problem into verification steps (checking r=1, r=2, then r=3), providing appropriately leveled support that decreases as the student progresses, and prompting self-correction rather than giving the answer directly when the student arrives at the correct solution.

---

## case_4 / constructive / pure_ai / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It looks like you tried to find the missing value by adding on the difference between the first two values. Let's try a different approach. Can you think about how the equation \( f=3r^{2}+3 \) relates to the values in the table and how you can use it to find the missing value?

**Turn 2 [student]**

> Oh right, so I need to use the equation itself? So like, I'd put in r = 3 into the equation f = 3r² + 3 to find the star? But I'm not really sure how to work that out - do I do 3 times 3 squared first, or...?

**Turn 3 [teacher]**

> That's a great start! Yes, you're on the right track by substituting \(r = 3\) into the equation. Remember, when evaluating an expression like \(3r^2\), you need to square the value of \(r\) first before multiplying by 3. Try working out \(3 \times 3^2\) step by step to find the value of \(f\) when \(r = 3\).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly names the misconception in Turn 1, redirects to the equation-based approach, and provides appropriately calibrated scaffolding that leads the student from linear thinking to correctly identifying the substitution method and asking about order of operations.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the constructive strategy by identifying the student's misconception, redirecting them to the correct approach, breaking down the order of operations into manageable steps, and adjusting support based on the student's expressed confusion about the calculation sequence.

---

## case_4 / constructive / pure_cbr_llm / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you may be treating the values in the table as linear, but remember that the equation given is \( f=3r^{2}+3 \). Try using this equation to calculate the missing value instead of adding on the differences between consecutive values in the table. Can you think of how to apply the equation to find the value covered by the star?

**Turn 2 [student]**

> Okay, I think I get what you're saying about using the equation instead of just adding. So, if I use \( r = 3 \) in the equation \( f = 3r^{2} + 3 \), I should be able to find the value for the star. But I'm not sure how to actually calculate it. Can you help me with that?

**Turn 3 [teacher]**

> Let's focus on how to calculate the value for the star when \( r = 3 \). Remember that the equation given is \( f = 3r^{2} + 3 \). Can you try substituting \( r = 3 \) into the equation and work through the calculation step by step to find the missing value?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 4 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly identifies the linear treatment misconception and redirects to the quadratic equation, maintaining cognitive demand by asking the student to work through the calculation, though the student's final turn shows understanding of the method but still requests help with execution, suggesting incomplete resolution of the underlying issue.

**Pass 2 (R6, persona-visible):** The teacher consistently scaffolds by breaking down the problem (identifying the misconception, then focusing on substitution), adjusts support based on student progress (moving from conceptual correction to procedural guidance), and stays within the student's ZPD by prompting rather than directly solving.

---

## case_4 / constructive / pure_cbr_llm / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> It seems like you're trying to find the missing value by adding on a pattern from the given numbers. Let's think about how we can use the equation \( f=3 r^{2}+3 \) to determine the value of \( f \) for a given \( r \) value. Can you walk me through your thought process on how you arrived at \( 24 \)?

**Turn 2 [student]**

> Yeah, so I looked at the first two numbers in the table - 6 and 15 - and the difference between them is 9, so I just kept adding 9 on each time. So 15 + 9 = 24. But I'm not really sure if that's what I'm supposed to be doing with the equation... should I be using the equation instead?

**Turn 3 [teacher]**

> It's great that you noticed the pattern in the table, but let's think about how the equation relates to those values. Can you think about how the equation \( f=3 r^{2}+3 \) could help us find the value of \( f \) for each \( r \) in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly identifies and addresses the linear pattern misconception, maintains cognitive demand through questioning, but the student shows only awareness of needing the equation without demonstrating actual corrected reasoning by the end.

**Pass 2 (R6, persona-visible):** The teacher exemplifies the constructive strategy by identifying the student's current understanding (pattern recognition), acknowledging it positively, then scaffolding toward the correct approach by breaking down the problem into manageable steps (understanding how to use the equation) without directly providing the answer.

---

## case_4 / constructive / pure_cbr_tpl / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I see what you're saying, but I'm still a bit confused. I thought I could just find the difference between the first two values and keep adding that to get the next one. So, if I got \( 6 \) to \( 15 \) by adding \( 9 \), I thought I could just add \( 9 \) again to get \( 24 \). Can you help me understand how to find the missing value correctly?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher completely ignores the student's clearly articulated misconception about linear patterns and instead repeatedly provides irrelevant examples from unrelated topics, offering no actual mathematical guidance on the quadratic function problem at hand.

**Pass 2 (R6, persona-visible):** The teacher provides no scaffolding, problem breakdown, or adjusted support—instead repeating identical, irrelevant examples from unrelated topics twice, completely ignoring the student's clearly articulated misconception about linear versus quadratic relationships.

---

## case_4 / constructive / pure_cbr_tpl / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so I see those examples are about different topics... but I'm still not really sure what I did wrong with my answer of 24. Did I just add the difference between 6 and 15 to get the next number? Is that not how you're supposed to do it with tables?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher completely ignores the student's explicitly stated misconception about adding differences in tables, instead repeatedly providing irrelevant examples from unrelated topics, resulting in no progress toward addressing the actual error with the quadratic function.

**Pass 2 (R6, persona-visible):** The teacher completely fails to implement the constructive strategy by repeatedly providing irrelevant examples from unrelated topics instead of scaffolding the student's understanding of the actual problem, ignoring the student's direct request for help understanding their specific error.

---

## case_4 / experiential / baseline / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you were trying to figure this out with objects you can touch and see. If you add the same amount each time, would it give you the right answer in this case?

**Turn 2 [student]**

> I guess if I think about it with objects, it might not work the same way every time. Like, if I just keep adding the same amount, it could be wrong. So, how do I actually find the right value for \( f \) when \( r = 3 \)?

**Turn 3 [teacher]**

> Imagine you were trying to figure this out with objects you can touch and see. If you add the same amount each time, would it give you the right answer in this case? Think about how the values in the table might relate to each other in a different way.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses the linear addition misconception indirectly through questioning but repeats the same prompt verbatim when the student asks for concrete help, failing to scaffold toward actually using the given equation, and the student shows awareness of the problem but no visible progress toward the correct method.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use concrete language like 'objects you can touch and see,' they fail to provide any actual real-world analogy or everyday situation that would help ground the abstract quadratic relationship in experiential understanding.

---

## case_4 / experiential / baseline / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this table as a real-life scenario. If you were trying to find the value covered by the star by looking at everyday objects, would adding on the difference between the first two values give the right answer?

**Turn 2 [student]**

> Hmm, I'm not really sure... I thought if the difference between 6 and 15 is 9, then I could just keep adding 9 to get 24 for the star. But now you're saying that might not work? Is it because the equation is more complicated than just adding the same number each time?

**Turn 3 [teacher]**

> Imagine if you were stacking blocks to represent the values in the table. Would adding on the same number each time still give you the correct height for the stack covered by the star?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague real-life analogies that don't directly address the linear vs. quadratic misconception, the student shows no progress in understanding how to actually use the given equation, and by Turn 3 the teacher simply repeats the same unhelpful analogy without moving toward resolution.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use concrete analogies (everyday objects, stacking blocks), these references remain vague and don't meaningfully connect the quadratic relationship to actual real-world experiences that would help the student understand why linear patterns fail here.

---

## case_4 / experiential / hybrid / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of the table as a journey. When you travel from \( r = 1 \) to \( r = 2 \), you don't just add the same amount each time. Each step is different, just like how each point in the table has its own unique value based on the equation. So, let's explore the equation \( f = 3r^{2} + 3 \) together to find the value covered by the star.

**Turn 2 [student]**

> Okay, I see what you mean about each step being different. So, if I use the equation for \( r = 3 \), I should plug that in instead of just adding like I did before? But how do I actually calculate it? Can you show me how to do it step by step?

**Turn 3 [teacher]**

> Let's think of plugging in \( r = 3 \) as picking a point in our journey. Just like coordinates help us find points on a line, using this value in the equation will help us find the corresponding value of \( f \). Let's walk through the steps together to uncover the value covered by the star in the table.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague metaphors about 'journeys' and 'coordinates' rather than directly addressing the linear addition misconception, never actually shows the calculation despite the student explicitly requesting it, and the student's final turn shows they still don't know how to perform the substitution.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies like 'journey' and 'coordinates,' these metaphors are vague and don't meaningfully connect the abstract mathematical operations to concrete everyday experiences that would help the student understand how to evaluate the quadratic function.

---

## case_4 / experiential / hybrid / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Think of the table as a map where each pair of values is like a coordinate. Just like on a map, you need both the x and y values to pinpoint a location. So, to find the missing value, you'll need to use the equation to calculate it based on the given coordinates. Can you see how this analogy helps clarify the process for finding the value covered by the star?

**Turn 2 [student]**

> Oh, I think I see what you mean... so I need to use the equation with the r value to find f, not just add on the difference like I did? So for r = 3, I'd put it into f = 3r² + 3 to get the answer? But I'm still a bit confused about how to actually work that out - do I do 3 times 3 squared first?

**Turn 3 [teacher]**

> Think of the equation as a set of directions on a treasure map. Just like you would follow the instructions step by step to reach the treasure, you need to follow the equation step by step to find the value of f. So, start by replacing r with 3 in the equation and then follow the order of operations to calculate the final answer. Can you see how this approach helps make sense of solving for the missing value in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 4 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses elaborate analogies (maps, treasure) rather than directly addressing the linear-addition misconception, but the student does recognize their error and shows improved understanding by Turn 2, asking appropriate follow-up questions about order of operations.

**Pass 2 (R6, persona-visible):** The teacher uses real-world analogies (maps with coordinates, treasure map directions) to explain abstract concepts, which aligns with the experiential strategy, but the analogies remain somewhat superficial and don't deeply ground the mathematical operations in genuinely concrete everyday experiences.

---

## case_4 / experiential / pure_ai / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of this table as a journey where each value of \( r \) represents a different stop. Instead of adding on the difference between the first two values, think about how you can use the equation \( f=3 r^{2}+3 \) to calculate the value for \( f \) at each stop along the way. Let's try approaching it from this new perspective.

**Turn 2 [student]**

> Okay, I think I get what you're saying about using the equation instead of just adding. So, if I plug in \( r = 3 \) into the equation \( f = 3 r^{2} + 3 \), I should be able to find the right value for \( f \). But what do I do after that? Do I just keep using the equation for the next value too?

**Turn 3 [teacher]**

> Let's think of each value of \( r \) as a different mile marker on our journey. Just like how you calculate the distance between two mile markers, think about how you can use the equation to find the distance to the next mile marker in our table. This way, you can calculate the value for \( f \) at each stop along the way.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses vague metaphors ('journey', 'mile markers') that don't directly address the linear addition misconception, and when the student shows understanding in Turn 2, the teacher repeats the same unhelpful metaphor instead of confirming the correct approach.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (journey, stops, mile markers), these metaphors don't meaningfully connect the abstract mathematical operations to concrete everyday experiences, and the 'distance between mile markers' analogy in Turn 3 actually confuses the task of direct substitution into an equation.

---

## case_4 / experiential / pure_ai / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you are at a carnival playing a game where you have to throw a ball at targets to win prizes. Each time you hit a target, the prize value increases by a certain amount. In this case, the equation is like the rules of the game, telling you how to calculate the prize value based on the target you hit. Let's think about how this analogy can help us find the value covered by the star in the table.

**Turn 2 [student]**

> Um, okay... so like, the equation is the rules of the game? But I'm still not really sure how that helps me find the star value. Are you saying I shouldn't just add on the difference like I did before (15 - 6 = 9, so 15 + 9 = 24)?

**Turn 3 [teacher]**

> Think about it this way: in the carnival game, each target hit increases the prize value by a specific amount, just like how the equation tells us to calculate the value based on the input. So, instead of just adding on the difference between the first two values, think about how the equation guides us to find the value covered by the star in a more systematic way. Let's try approaching it from that perspective.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses a vague carnival analogy that doesn't address the linear-addition misconception directly, provides no concrete guidance on using the quadratic equation, and the student shows no improvement—explicitly asking if they should stop adding differences but receiving only another abstract restatement.

**Pass 2 (R6, persona-visible):** While the teacher introduces a carnival game analogy, it remains vague and disconnected from the mathematical structure of the quadratic function, failing to meaningfully ground the abstract equation in concrete experience or help the student understand how to actually apply f=3r²+3.

---

## case_4 / experiential / pure_cbr_llm / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think of the table as a map where each pair of coordinates guides us to a specific location. Just like using GPS coordinates to find a hidden treasure, we need to use the equation to guide us to the correct value covered by the star in the table.

**Turn 2 [student]**

> I get what you're saying about using the equation like a map, but I'm still a bit confused. I thought I could just keep adding the differences like I did before. So, if I use the first two values, shouldn't I just keep adding that difference to find the next one? How do I use the equation instead?

**Turn 3 [teacher]**

> Let's think of the equation as a secret code that unlocks the value covered by the star. Just like cracking a code in a spy mission, we need to use the equation \( f=3 r^{2}+3 \) to decipher the correct value. Remember, each pair of coordinates holds a clue to help us solve the mystery.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 3 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly uses vague metaphors (GPS, treasure, spy mission) without addressing the student's explicit misconception about adding differences in linear patterns, never guides the student to actually substitute r=3 into the equation, and the student shows no improvement—in fact explicitly restating their confusion about whether to add differences or use the equation.

**Pass 2 (R6, persona-visible):** While the teacher attempts to use real-world analogies (GPS, treasure maps, spy missions), these metaphors are superficial and don't actually ground the mathematical concept in concrete everyday experiences that help the student understand how to apply the equation.

---

## case_4 / experiential / pure_cbr_llm / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Imagine you have a pair of coordinates on a map, like a treasure hunt. The coordinates tell you exactly where the treasure is hidden. In this case, the equation is like a map, and the values in the table are like the coordinates. Can you see how you might use the equation to guide you to the correct value for the star in the table?

**Turn 2 [student]**

> Oh, so you're saying I should use the equation itself to find the answer, not just look at the pattern in the numbers? So like, I'd put r = 3 into the equation f = 3r² + 3 to find what f is? But I'm still a bit confused about how that's different from what I was doing before...

**Turn 3 [teacher]**

> Imagine you're looking at a line graph showing how many people are in a cafe at different times of the day. If you want to know how many people were there at 3 PM, you wouldn't just guess based on the other points, right? You'd use the graph to find the exact number. Can you see how using the equation in this table is like using the graph to find the value for the star?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher uses metaphors rather than directly addressing the linear-difference misconception, and while the student shows some progress in Turn 2 by recognizing they should substitute into the equation, they remain confused about why their original method was wrong, and the teacher's Turn 3 response doesn't clarify this distinction.

**Pass 2 (R6, persona-visible):** The teacher consistently grounds the abstract mathematical concept of function evaluation in concrete, relatable real-world scenarios (treasure hunt with map coordinates, reading a cafe attendance graph), which is a textbook application of the experiential strategy.

---

## case_4 / experiential / pure_cbr_tpl / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I think I see what you're saying about not just adding the difference between the first two values. But I'm still a bit confused about how to actually find the missing value for \( f \) when \( r = 3 \). Can you help me understand how to use the equation \( f = 3r^2 + 3 \) to get the right answer?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples from unrelated topics instead of addressing the student's misconception about linear patterns or helping them substitute r=3 into the given quadratic equation, resulting in no meaningful progress.

**Pass 2 (R6, persona-visible):** The teacher provides no concrete real-world analogies or everyday situations whatsoever, instead repeatedly listing abstract mathematical topics and misconceptions without any experiential grounding.

---

## case_4 / experiential / pure_cbr_tpl / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Um, okay... so I see those examples are about different things like lines and decimals, but I'm still not really sure how that helps with my problem? Like, I found the difference between the first two f values (15 - 6 = 9) and then added that on again to get 24, but are you saying that's not how you're supposed to do it with this equation?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides completely irrelevant examples about unrelated topics (linear equations, decimal subtraction, line graphs) that have nothing to do with the student's quadratic function problem, failing to address the misconception of treating a quadratic relationship as linear, providing no appropriate scaffolding, and leaving the student confused as evidenced by their response.

**Pass 2 (R6, persona-visible):** The teacher provides no real-world analogies or concrete everyday situations, instead repeatedly presenting abstract mathematical examples from unrelated topics that don't connect to the student's quadratic function problem.

---

## case_4 / rule_based / baseline / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> There is a specific procedure for finding values in the table. Step 1 is to identify the operation being used in the equation, and step 2 is to apply that operation to the given values. In this case, you should plug in the values of \( r \) into the equation to find the corresponding values of \( f \).

**Turn 2 [student]**

> Okay, I think I get what you're saying about plugging in the values for \( r \) into the equation. So, for \( r = 1 \), I got \( f = 6 \), and for \( r = 2 \), I got \( f = 15 \). But I'm still a bit confused about how to find the value for \( r = 3 \). Do I just keep adding the difference like I did before, or is there another way?

**Turn 3 [teacher]**

> There is a specific procedure for finding values in the table. Step 1 is to identify the operation being used in the equation, and step 2 is to apply that operation to the given values. In this case, you should plug in the value of \( r = 3 \) into the equation to find the corresponding value of \( f \).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeatedly gives generic procedural instructions without addressing the student's linear-addition misconception, provides no scaffolding to help the student actually perform the substitution, and the student explicitly states they're still confused about whether to keep adding differences, showing no improvement in understanding.

**Pass 2 (R6, persona-visible):** The teacher states explicit procedures (identifying operations and plugging in values) but never shows worked steps, provides a worked example, or gives direct corrective feedback on the student's specific procedural error that led to the answer of 24.

---

## case_4 / rule_based / baseline / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> There is a specific procedure for finding values in a table. Step 1 is to identify the operation being used in the equation. Step 2 is to apply the rule for that operation. The mistake you made was not following the correct procedure for this problem.

**Turn 2 [student]**

> Oh right, so I need to actually use the equation instead of just looking at the pattern in the numbers? So like, I should plug in r = 3 into the equation f = 3r² + 3 to find the star value, rather than just adding the difference like I did?

**Turn 3 [teacher]**

> Yes, that's correct. You should plug in \( r = 3 \) into the equation \( f = 3r^2 + 3 \) to find the value of the star, rather than just adding the difference between the first two values. Remember to follow the specific procedure for this type of problem.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides generic procedural advice without engaging the specific linear-pattern misconception, demands no reasoning from the student, but the student nonetheless self-corrects by articulating the difference between using the equation versus looking at patterns.

**Pass 2 (R6, persona-visible):** The teacher states explicit procedures and gives direct corrective feedback, but fails to show worked steps or demonstrate the actual calculation (3(3)² + 3 = 30), which is a key component of rule-based instruction.

---

## case_4 / rule_based / hybrid / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> The correct approach to find the missing value in the table is to substitute the given values of \( r \) into the equation \( f=3r^{2}+3 \). You should not add on the difference between the first two values. Let's walk through the correct calculation together step by step.

**Turn 2 [student]**

> Okay, I see what you mean about not just adding the difference. So, if I plug in the values for \( r \) into the equation, I should be able to find \( f \) directly. Can you show me how to do that for \( r = 3 \)? I'm still a bit confused about how to use the equation properly.

**Turn 3 [teacher]**

> Yes, that's correct. To find the value of \( f \) when \( r = 3 \), substitute \( r = 3 \) into the equation \( f = 3r^{2} + 3 \). Let's work through the steps together to calculate the correct value of \( f \) for \( r = 3 \).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly names the misconception (adding differences) and directs toward substitution, but repeatedly tells rather than elicits reasoning, and the student remains dependent on being shown rather than demonstrating independent understanding of how to apply the equation.

**Pass 2 (R6, persona-visible):** While the teacher identifies the correct procedure and mentions working through steps, they never actually state the explicit steps, show worked calculations, or provide the direct answer, instead repeatedly promising to do so without following through.

---

## case_4 / rule_based / hybrid / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> You cannot find the missing value by adding on the difference between the first two values. To correctly find the value covered by the star, you need to substitute the given value of \( r \) into the equation \( f=3r^{2}+3 \). Let's walk through the steps together to find the correct answer.

**Turn 2 [student]**

> Oh right, so I need to use the equation instead of just adding on? So for the star, I'd substitute r = 3 into f = 3r² + 3... so that would be f = 3 times 3 squared plus 3? Can you help me work out what 3 squared is?

**Turn 3 [teacher]**

> That's correct. To find the value covered by the star, you should substitute \( r = 3 \) into the equation \( f = 3r^{2} + 3 \). When you calculate \( 3^{2} \), you get \( 9 \), so the correct calculation would be \( f = 3 \times 9 + 3 \). Can you now determine the value of \( f \) for \( r = 3 \)?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly addresses the linear addition misconception and the student shows clear progress in understanding the substitution method, though the teacher provides slightly more help than needed given the student's demonstrated capability in Turn 2.

**Pass 2 (R6, persona-visible):** The teacher consistently provides explicit procedural instructions (substitute r into the equation), gives direct corrective feedback on the student's error (cannot find by adding differences), shows worked steps (breaking down the calculation of 3² = 9, then 3 × 9 + 3), which perfectly exemplifies rule-based teaching.

---

## case_4 / rule_based / pure_ai / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To find the value covered by the star, you need to substitute the given values of \( r \) into the equation \( f=3r^{2}+3 \). Let's check your work using this method. Let's start by substituting \( r=3 \) into the equation to find the correct value.

**Turn 2 [student]**

> Okay, so if I put \( r=3 \) into the equation \( f=3r^{2}+3 \), I think I would do \( 3 \times 3^{2} + 3 \). So that’s \( 3 \times 9 + 3 \), which is \( 27 + 3 = 30 \). Is that right? But I thought I could just keep adding the differences like I did before. Why does it change?

**Turn 3 [teacher]**

> You made a mistake in your calculation. When substituting \( r=3 \) into the equation \( f=3r^{2}+3 \), you should get \( 3 \times 3^{2} + 3 = 3 \times 9 + 3 = 27 + 3 = 30 \). The correct method involves using the equation to find the values, not just adding on differences between values in the table. Remember to follow the rule of substitution for each \( r \) value.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 2 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher tells the student to use substitution but never addresses why the linear difference method fails, the student correctly calculates 30 but the teacher incorrectly says this is wrong, and the student's final question about why differences don't work remains unanswered, showing no resolution of the misconception.

**Pass 2 (R6, persona-visible):** The teacher consistently states explicit procedures (substitution method), shows worked steps (3 × 3² + 3 = 3 × 9 + 3 = 27 + 3 = 30), and gives direct corrective feedback on the procedural error of using differences instead of substitution, which is a textbook example of rule-based teaching.

---

## case_4 / rule_based / pure_ai / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> To find the value covered by the star, you need to substitute \( r=3 \) into the equation \( f=3r^{2}+3 \). Let's correct your calculation by plugging in \( r=3 \) step by step.

**Turn 2 [student]**

> Oh right, so I need to use the equation itself, not just add on the difference? So if I put in r=3, I get f = 3 times 3 squared plus 3... so that's 3 times 9 plus 3, which is 27 plus 3 = 30?

**Turn 3 [teacher]**

> Actually, you made a mistake in your calculation. When you substitute \( r=3 \) into the equation, you should have \( f=3(3)^{2}+3 \), not \( f=3\times 3+3 \). Let's correct that step and find the correct value for the star in the table.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher addresses the linear addition misconception in Turn 1, but Turn 3 contains a significant mathematical error (claiming the student wrote 3×3+3 when they actually wrote 3×9+3, which was correct), though the student does show improvement by abandoning the linear approach and correctly calculating the answer in Turn 2.

**Pass 2 (R6, persona-visible):** While the teacher identifies procedural errors and attempts to give corrective feedback, they fail to actually show the worked steps or state the explicit procedure to completion, leaving the student without a demonstrated solution.

---

## case_4 / rule_based / pure_cbr_llm / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> You made a procedural error in finding the missing value in the table. To correct this, remember that for the equation \( f=3 r^{2}+3 \), you need to substitute the given value of \( r \) into the equation and solve for \( f \). Try this method again to find the correct value covered by the star.

**Turn 2 [student]**

> Okay, I think I get what you're saying about using the equation. So, for \( r = 3 \), I should plug that into the equation \( f = 3(3^2) + 3 \). But I'm not sure how to do the calculations properly. Can you help me with that part?

**Turn 3 [teacher]**

> You made a procedural error in your calculation. Remember to correctly follow the order of operations when substituting \( r = 3 \) into the equation. Start by calculating \( 3(3^2) \) first, then add 3 to find the correct value for \( f \).

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never addresses the student's specific misconception about treating the row as linear and adding differences, instead only providing generic procedural corrections, and the student shows minimal reasoning improvement by Turn 2, merely asking for calculation help rather than demonstrating understanding.

**Pass 2 (R6, persona-visible):** The teacher provides direct corrective feedback on procedural errors and mentions the order of operations, but stops short of showing worked steps or stating the complete explicit procedure that would exemplify a full rule-based approach.

---

## case_4 / rule_based / pure_cbr_llm / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> You made an error in your calculation. The correct approach is to substitute the given values of \( r \) into the equation \( f=3 r^{2}+3 \) to find the corresponding \( f \) values. Let's go through the correct steps together to find the value covered by the star.

**Turn 2 [student]**

> Oh right, so I need to use the equation itself rather than just looking at the pattern in the numbers? So like, I substitute r = 3 into f = 3r² + 3? Can you show me how to do that bit because I'm not sure what substituting means exactly.

**Turn 3 [teacher]**

> Yes, that's correct. To substitute a value means to replace the variable with the given number. In this case, substitute \( r = 3 \) into the equation \( f = 3r^2 + 3 \) and work out the calculation step by step. Let's do that now.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher correctly identifies an error and provides mathematically accurate guidance, but never explicitly addresses the student's linear pattern misconception, offers minimal cognitive challenge by directing rather than probing, and while the student shows some progress in understanding they need to use the equation, they still require basic procedural help with substitution.

**Pass 2 (R6, persona-visible):** While the teacher identifies procedural errors and mentions working through steps, they consistently stop short of actually showing the worked steps or stating the explicit procedure, instead prompting the student to do it themselves, which is inconsistent with the rule-based strategy's emphasis on demonstrating procedures.

---

## case_4 / rule_based / pure_cbr_tpl / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I see what you mean about using the equation to find the missing value instead of just adding. But I'm still a bit confused about how to actually use the equation \( f=3 r^{2}+3 \) to find \( f \) when \( r = 3 \). Can you help me understand that part better?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples from unrelated topics instead of addressing the student's misconception about linear patterns or helping them substitute r=3 into the quadratic equation, resulting in no meaningful progress.

**Pass 2 (R6, persona-visible):** The teacher never states explicit procedures, shows worked steps, or gives direct corrective feedback; instead, they only present irrelevant examples from other topics without addressing the student's specific question about how to substitute r=3 into the equation.

---

## case_4 / rule_based / pure_cbr_tpl / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Um, okay... so I see those examples are about different topics, but I'm still not really sure what I did wrong with my question. I added the difference between 6 and 15 (which is 9) to get 24, but is that not how you're supposed to do it with tables?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher completely ignores the student's clearly articulated misconception about adding constant differences in tables, instead repeatedly providing irrelevant examples from unrelated topics, resulting in no progress toward addressing the actual error with quadratic functions.

**Pass 2 (R6, persona-visible):** The teacher provides no explicit procedures, worked steps, or direct corrective feedback, instead repeatedly presenting irrelevant examples from unrelated problems without addressing the student's specific procedural error.

---

## case_4 / socratic / baseline / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> What would happen if you tried your method on a simpler version of the problem?

**Turn 2 [student]**

> I guess I could try it with just \( r = 1 \) and \( r = 2 \) to see if it works. So if \( f(1) = 6 \) and \( f(2) = 15 \), I would just add the difference, which is \( 9 \), to \( 15 \) to get \( 24 \) again. But I'm not sure if that's how it's supposed to work. Should I be using the equation instead?

**Turn 3 [teacher]**

> What do you notice about the values in the table as the input \( r \) increases by 1 each time?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively engages the misconception by prompting the student to examine the pattern of differences (which aren't constant in quadratic relationships), but the student's final turn shows they haven't yet moved beyond their linear thinking, only expressing uncertainty rather than demonstrating improved reasoning.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking without directly providing the answer, and implicitly challenges the student's flawed pattern-based reasoning by directing attention to examine the relationship between inputs and outputs more carefully.

---

## case_4 / socratic / baseline / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> What would happen if you tried your method on a simpler version of the problem?

**Turn 2 [student]**

> Um, okay... so like, if I look at the first two values, the difference between 6 and 15 is 9, right? So I just kept adding 9 to get 24 for the star. But I'm not really sure what you mean by a simpler version - do you mean like, using smaller numbers or something?

**Turn 3 [teacher]**

> Can you think of a different way to approach the problem that doesn't involve just adding a constant difference between the first two values?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks vague questions without directly addressing the misconception that this is a quadratic (not linear) relationship, and the student shows no improvement in understanding, merely restating their linear approach and expressing confusion about what's being asked.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward recognizing the flaw in their linear pattern assumption without directly providing the answer or the correct method of substituting r=3 into the equation.

---

## case_4 / socratic / hybrid / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Why do you think adding the difference between the first two values would give you the correct value for the star in this table? Can you explain how the equation \( f=3r^{2}+3 \) relates to the values in the table?

**Turn 2 [student]**

> I thought that since the first two values of \( f \) are \( 6 \) and \( 15 \), I could just find the difference, which is \( 9 \), and then add that to \( 15 \) to get \( 24 \). I guess I didn't really think about how the equation works with the values of \( r \). Can you help me understand how to use the equation to find the missing value instead?

**Turn 3 [teacher]**

> Why do you think the equation \( f=3r^{2}+3 \) involves squaring the value of \( r \) and then multiplying by 3 before adding 3? How might this information help you find the missing value in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively identifies and addresses the linear thinking misconception through questioning, maintains high cognitive demand by asking the student to reason about the quadratic structure, but the student never actually demonstrates corrected understanding by computing or stating the correct answer.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never provides the answer directly, and in Turn 1 implicitly challenges the flawed reasoning of using linear differences for a quadratic function, which is textbook Socratic method.

---

## case_4 / socratic / hybrid / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's take a step back and think about how we can approach this problem differently. How can we use the equation \( f=3r^2+3 \) to help us find the missing value in the table? Can we use the given values in the table to guide us in a different direction?

**Turn 2 [student]**

> Oh, so you mean I should use the equation itself instead of just looking at the pattern in the numbers? Like, I need to plug in the value of r into the equation f=3r²+3 to find f? So for r=3, I'd do 3 times 3 squared plus 3?

**Turn 3 [teacher]**

> Great observation! Now, let's think about how we can use the equation to find the value of \( f \) for \( r = 3 \). Can you walk me through the steps of plugging in \( r = 3 \) into the equation \( f = 3r^2 + 3 \) and what the result would be?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never explicitly addresses the linear pattern misconception, but the student independently realizes they should use the equation instead of patterns, showing good trajectory improvement despite minimal scaffolding specificity.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward the correct approach without ever directly providing the answer, which is a textbook example of the Socratic method.

---

## case_4 / socratic / pure_ai / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you explain to me why you think adding on the difference between the first two values will give you the correct answer for the missing value in the table? Let's think about how the equation \( f=3r^{2}+3 \) relates to the values in the table and how we can use it to find the missing value covered by the star.

**Turn 2 [student]**

> I thought that since the first two values for \( f \) were \( 6 \) and \( 15 \), I could just find the difference, which is \( 9 \), and then add that to \( 15 \) to get \( 24 \). But now I'm wondering if the equation actually gives a different way to find \( f \) for \( r = 3 \). Can you help me understand how to use the equation properly?

**Turn 3 [teacher]**

> That's a great observation! Let's explore how the equation relates to the values in the table for \( r = 3 \). Can you think about how plugging in \( r = 3 \) into the equation \( f=3r^{2}+3 \) might help us find the value for \( f \) at \( r = 3 \)?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 3 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 3.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly addresses the linear-addition misconception and the student demonstrates clear progress in recognizing the need to use the equation, though the teacher's questions become increasingly leading rather than demanding deeper reasoning.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student toward the correct approach, never directly provides the answer, and in Turn 1 implicitly challenges the flawed reasoning about using differences between consecutive values, which are all hallmarks of the Socratic method.

---

## case_4 / socratic / pure_ai / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> How did you decide to add on the difference between the first two values? Can you explain why you think that method is correct?

**Turn 2 [student]**

> Well, I looked at the first two values in the f row - 6 and 15 - and the difference is 9, so I thought if I keep adding 9 each time it would work like a pattern. But now you're asking me that, I'm not sure... is that not how you're supposed to do it? Should I be using the equation instead?

**Turn 3 [teacher]**

> Can you think of a reason why adding on the difference between the first two values might not work for this specific table? What other information can you gather from the equation provided that could help you find the missing value in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively addresses the linear addition misconception by prompting the student to question their method and consider the given equation, leading the student to recognize they should use the equation instead of assuming a constant difference pattern.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never directly provides the answer, and effectively exposes the flaw in the student's linear pattern assumption by prompting them to reconsider their method against the given quadratic equation.

---

## case_4 / socratic / pure_cbr_llm / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Can you walk me through your thought process on how you arrived at the value of 24? What assumptions did you make when approaching this problem?

**Turn 2 [student]**

> I thought that since the first two values in the table are 6 and 15, I could just find the difference between them, which is 9, and then add that to 15 to get the next value. So, I added 9 to 15 and got 24. But now I'm not sure if that's the right way to do it. Should I be using the equation instead?

**Turn 3 [teacher]**

> Can you explain why you chose to add the difference between the first two values to find the missing value in the table? Is there another approach you could consider that might align more closely with the equation given \( f=3 r^{2}+3 \)?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 4 |
| R3 scaffolding fit | 4 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 4.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher effectively identifies and addresses the linear addition misconception through questioning, maintains high cognitive demand by asking students to explain their reasoning, but we don't see evidence of improved student understanding since the dialogue ends before the student demonstrates corrected thinking.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never provides the answer directly, and subtly exposes the flaw in the student's pattern-based reasoning by redirecting attention to the given equation.

---

## case_4 / socratic / pure_cbr_llm / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Why did you choose to add 3 to the previous value instead of following a different approach? Can you think of a different method that might lead to the correct answer?

**Turn 2 [student]**

> Um, I added 9 to get from 6 to 15, so I thought I'd add 9 again to get 24... but I'm not really sure why that would work? Is there a different way to find the answer, like using the equation they gave us at the start?

**Turn 3 [teacher]**

> Why do you think adding 9 twice would give you the correct answer? Can you explain how the equation given relates to the values in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 5 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher asks questions requiring reasoning but never directly addresses why the linear pattern assumption fails for quadratic relationships, and while the student begins to question their method and mentions the equation, they don't demonstrate clear understanding of how to use it correctly by the end.

**Pass 2 (R6, persona-visible):** The teacher consistently uses probing questions to guide the student's thinking, never provides the answer directly, and challenges the student's flawed pattern-based reasoning by prompting them to reconsider their approach and connect back to the given equation.

---

## case_4 / socratic / pure_cbr_tpl / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I see what you're saying, but I'm still a bit confused. I thought I could just add the difference between the first two \( f \) values to find the missing one. So, if \( 15 - 6 = 9 \) and then I add that to \( 15 \), I thought it would be \( 24 \). Can you help me understand how to actually use the equation instead?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides completely irrelevant examples about unrelated topics (linear equations, decimal subtraction, line graphs) that have nothing to do with the student's quadratic function misconception, offering no actual engagement with the student's explicit request for help understanding how to use the given equation.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions, provide counterexamples, or guide discovery; instead, they repeatedly paste irrelevant examples from unrelated math problems that don't address the student's specific misconception about quadratic functions.

---

## case_4 / socratic / pure_cbr_tpl / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying I shouldn't just add the difference between the first two numbers? I think I see what you mean - like, I found that 15 - 6 = 9, so I added 9 again to get 24, but that's not how it actually works with this equation, is it? Should I be using the equation f = 3r² + 3 to work it out instead?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides only irrelevant examples from unrelated topics instead of addressing the student's linear-addition misconception, offering no actual guidance despite the student explicitly recognizing their error and asking for confirmation about using the given equation.

**Pass 2 (R6, persona-visible):** The teacher does not ask any probing questions or use counterexamples; instead, they repeatedly paste irrelevant examples from unrelated math topics, completely failing to engage with the student's reasoning or guide them through questioning.

---

## case_4 / traditional / baseline / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me explain the correct method, and then we will work through an example together. In this case, we need to substitute the values of \( r \) into the equation \( f=3 r^{2}+3 \) to find the corresponding values of \( f \). Let's start with \( r=3 \) and solve for \( f \).

**Turn 2 [student]**

> Okay, so I think I need to plug in \( r=3 \) into the equation \( f=3 r^{2}+3 \). So, would that mean I do \( 3 \times 3^{2} + 3 \)? But I'm not really sure how to do that step properly. Can you help me with the calculation?

**Turn 3 [teacher]**

> Let me explain the correct method, and then we will work through an example together. To find the value of \( f \) when \( r=3 \), you need to substitute \( 3 \) into the equation \( f=3 r^{2}+3 \) and perform the calculations step by step. Let's work through it together.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never addresses the student's linear pattern misconception (adding 9 repeatedly), provides correct mathematics but with minimal cognitive demand, and the student shows no improvement—still asking for help with basic calculation at the end.

**Pass 2 (R6, persona-visible):** While the teacher attempts to explain the method and work through examples, they repeat the same general statement without actually providing the worked example or step-by-step calculation that is central to traditional instruction, failing to demonstrate the computational steps the student explicitly requested.

---

## case_4 / traditional / baseline / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let me explain the correct method, and then we will work through an example together. In this case, we need to substitute the given values of \( r \) into the equation \( f=3 r^{2}+3 \) to find the missing value. Let's work through it step by step.

**Turn 2 [student]**

> Oh right, so I need to use the equation itself rather than just looking at the pattern in the numbers? So like, I'd put in r = 3 into the equation f = 3r² + 3 to find the star? Can you show me how to do that bit because I'm not sure what the r² means exactly.

**Turn 3 [teacher]**

> Let me explain the correct method, and then we will work through an example together. In this case, we substitute the value of \( r \) into the equation \( f=3 r^{2}+3 \) to find the missing value. Let's work through it step by step.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides mathematically correct guidance but fails to address the student's linear pattern misconception, offers no cognitive challenge by simply stating the method rather than eliciting reasoning, and responds to a specific question about r² with an identical scripted response, leaving the student's confusion unresolved.

**Pass 2 (R6, persona-visible):** While the teacher announces intent to explain and work through examples (traditional approach), they fail to actually provide the worked example, direct explanation of r², or demonstrate the calculation despite the student's explicit request for clarification.

---

## case_4 / traditional / hybrid / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's revisit the concept of finding values in a table using the equation \( f=3 r^{2}+3 \). Remember, each value of \( r \) corresponds to a specific value of \( f \) based on the equation. Can you explain how you approached finding the value covered by the star in the table?

**Turn 2 [student]**

> I thought I could just keep adding the difference between the first two values, which is \( 15 - 6 = 9 \), to get to the next one. So I added \( 9 \) to \( 15 \) and got \( 24 \). But now I'm not sure if that's the right way to do it. Should I be using the equation for each value of \( r \) instead?

**Turn 3 [teacher]**

> You're on the right track by recognizing that each value of \( r \) corresponds to a specific value of \( f \) based on the equation \( f=3 r^{2}+3 \). Instead of adding the difference between the first two values, try plugging in each value of \( r \) into the equation to find the corresponding value of \( f \). Can you try using the equation to calculate the value covered by the star in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 3 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher acknowledges the linear addition misconception and redirects to using the equation, but provides minimal scaffolding for why the linear approach fails; the student self-corrects by Turn 2, suggesting the misconception was already weakening before substantive teaching occurred.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning and elicits student thinking rather than directly explaining the concept with a worked example, which is inconsistent with the traditional strategy of direct explanation.

---

## case_4 / traditional / hybrid / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's revisit the approach to finding missing values in a table using the equation given. It seems like you may be applying a linear pattern incorrectly. Can you explain your thought process in finding the missing value, and we can work through it together?

**Turn 2 [student]**

> Yeah, so I looked at the first two values in the f row - 6 and 15 - and the difference between them is 9, so I thought I'd just keep adding 9 each time, which gave me 24 for the star. But you're saying that's not right because... is it something to do with the equation they gave us at the top? Should I be using that instead?

**Turn 3 [teacher]**

> That's a good observation about the pattern between the first two values, but in this case, we need to use the equation provided to find the missing value. Let's focus on how the equation \( f=3 r^{2}+3 \) can help us determine the value at the star in the table. Can you think about how we can use this equation to solve for the missing value?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 4 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.67 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher explicitly identifies the linear pattern misconception and redirects to the equation, but asks overly open questions that don't scaffold the student toward actually using the equation, and we see no evidence the student performs the calculation or corrects their thinking by the end.

**Pass 2 (R6, persona-visible):** The teacher uses questioning and guided discovery rather than direct explanation with worked examples, which is inconsistent with the traditional strategy of explicitly demonstrating the solution method.

---

## case_4 / traditional / pure_ai / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's take a step back and examine the pattern in the table. Remember, to find the value covered by the star, we need to substitute \( r \) into the equation \( f=3r^{2}+3 \) and solve for \( f \). Can you try that approach and see if you get a different answer than \( 24 \)?

**Turn 2 [student]**

> Okay, I see what you mean about using the equation directly. So, if I put \( r = 3 \) into the equation \( f = 3r^{2} + 3 \), I think I should calculate it like this: \( f = 3(3^{2}) + 3 \). But I'm not sure how to do that step properly. Can you help me with the calculation?

**Turn 3 [teacher]**

> Let's break down the calculation step by step. First, square the value of \( r \) to get \( 3^{2} = 9 \). Then multiply that result by 3 to get \( 3(9) = 27 \). Finally, add 3 to that result to find the value of \( f \) when \( r = 3 \). Can you try these calculations and see if you get the correct answer for the value covered by the star in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 2 |
| R6 strategy fidelity | 3 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never explicitly addresses the linear pattern misconception, provides overly detailed scaffolding in Turn 3 that reduces cognitive demand, and the student shows no evidence of understanding why their original linear approach was incorrect.

**Pass 2 (R6, persona-visible):** The teacher provides step-by-step guidance and breaks down the calculation clearly, which aligns with traditional methods, but relies heavily on prompting the student to perform calculations rather than demonstrating a complete worked example first.

---

## case_4 / traditional / pure_ai / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's go over the correct method for finding values in the table using the equation provided. Remember, each value in the table must satisfy the equation. Can you explain how you arrived at \(24\) and we can work through it together?

**Turn 2 [student]**

> Um, yeah... so I looked at the first two numbers in the f row - that's 6 and 15 - and the difference between them is 9, so I just added 9 again to get 24. But I'm guessing that's not right because you're saying we need to use the equation? Should I be plugging in the r value into the equation instead?

**Turn 3 [teacher]**

> Yes, you're on the right track by recognizing the pattern in the differences between the values, but to accurately find the missing value, you need to substitute the given value of \( r \) into the equation \( f=3r^2+3 \) and solve for \( f \). Can you try plugging in \( r=3 \) and see what you get for \( f \)?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 4 |
| R6 strategy fidelity | 2 |
| Quality composite (R1+R2+R3)/3 | 2.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never explicitly addresses the linear-addition misconception that the student clearly articulates in Turn 2, though the student does self-correct by recognizing they should use the equation instead, showing improvement in reasoning by the end.

**Pass 2 (R6, persona-visible):** The teacher uses guided questioning and elicits student thinking rather than directly explaining the method with a worked example, which is inconsistent with the traditional strategy of direct explanation.

---

## case_4 / traditional / pure_cbr_llm / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's take a step back and think about how we can find the value covered by the star in the table. Remember, the equation given is \( f=3 r^{2}+3 \). How can we use this equation to correctly determine the missing value in the table?

**Turn 2 [student]**

> I think I need to use the equation \( f=3 r^{2}+3 \) for each value of \( r \) to find \( f \). So for \( r=3 \), I guess I should plug it into the equation instead of just adding like I did before. But I'm not really sure how to do that. Can you help me figure out what to do next?

**Turn 3 [teacher]**

> Let's focus on understanding how to correctly use the equation \( f=3 r^{2}+3 \) to find the missing value for \( r=3 \). Remember, we need to substitute \( r=3 \) into the equation and follow the correct steps to determine the corresponding value of \( f \). Can you walk me through your thought process as you work through this calculation?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 2 |
| R2 cognitive demand | 3 |
| R3 scaffolding fit | 2 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 3 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 2.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher never explicitly addresses the linear addition misconception, provides generic prompts about using the equation, and while the student recognizes they should substitute rather than add, they remain uncertain about execution and the dialogue ends before demonstrating successful correction.

**Pass 2 (R6, persona-visible):** The teacher consistently uses questioning and prompting to elicit student thinking rather than directly explaining the procedure or providing a worked example, which is contrary to the traditional strategy of direct explanation.

---

## case_4 / traditional / pure_cbr_llm / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's address your misconception. Remember, in this case, we need to use the equation \( f = 3r^2 + 3 \) to find the missing value. Can you explain how you applied this equation to find the value covered by the star in the table?

**Turn 2 [student]**

> Um, I think I just looked at the difference between the first two numbers - so 15 minus 6 is 9, and then I added 9 to 15 to get 24. But now you're saying I need to use the equation instead? So do I put r = 3 into the equation f = 3r² + 3?

**Turn 3 [teacher]**

> Let's address your misconception. Remember, we need to use the equation \( f = 3r^2 + 3 \) to find the missing value. Can you explain how you applied this equation to find the value covered by the star in the table?

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 2 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 5 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.33 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher repeats identical generic prompts without addressing the student's explicitly stated linear-difference misconception, and the student shows no improvement as the dialogue ends with the same question being asked twice.

**Pass 2 (R6, persona-visible):** The teacher fails to provide direct explanation or worked examples characteristic of traditional teaching, instead repeatedly asking the student to explain their work without offering the clear, comprehensive instruction that defines the traditional approach.

---

## case_4 / traditional / pure_cbr_tpl / leg_a

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> I see what you're saying, but I'm still a bit confused. I thought I could just find the difference between the first two values and keep adding that to get the next one. So, if I got \( 6 \) to \( 15 \) by adding \( 9 \), I thought I could just add \( 9 \) again to get \( 24 \). But is that not how it works? How do I actually find the missing value using the equation?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher completely ignores the student's clearly articulated misconception about linear patterns and instead repeatedly provides irrelevant examples from unrelated topics, offering no actual mathematical guidance on the quadratic function problem at hand.

**Pass 2 (R6, persona-visible):** The teacher does not explain anything directly, provide worked examples, or check understanding; instead, they repeatedly paste irrelevant examples from unrelated topics that don't address the student's quadratic function question.

---

## case_4 / traditional / pure_cbr_tpl / leg_b

**Misconception:** Thinks you can find missing values in a given table by treating the row as linear and adding on the difference between the first two values given.

**Judge:** anthropic (model: claude-sonnet-4-5)

### Turns

**Turn 0 [student]**

> I have: The equation \( f=3 r^{2}+3 \) is used to find values in the table below. What is the value covered by the star? \begin{tabular}{|c|c|c|c|c|} \hline\( r \) & \( 1 \) & \( 2 \) & \( 3 \) & \( 4 \) \\ \hline\( f \) & \( 6 \) & \( 15 \) & \( \color{gold}\bigstar \) & \\ \hline \end{tabular}. I got \( 24 \) because I'm not sure I worked it out the right way.

**Turn 1 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

**Turn 2 [student]**

> Oh right, so you're saying I shouldn't just add on the difference like I did? I found the difference between 6 and 15, which is 9, and then added 9 to 15 to get 24... but I'm confused about what I should actually be doing instead - should I be using the equation somehow?

**Turn 3 [teacher]**

> Let's think about this. From similar cases:
>   1. Topic: Given a positive x value, find a y value in a table of values for lines in the form y = mx + c. Correct answer (C): \( 1 \). Student selected: \( 6 \). Underlying misconception: Does not realise that a pair of coordinates can be used to find missing parts of an equation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   2. Topic: Subtract decimals where the numbers involved have a different number of decimal places. Correct answer (C): \( 1.62 \). Student selected: \( 4.41 \). Underlying misconception: Hasn't lined up the place value columns when writing out the calculation. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.
>   3. Topic: Interpret continuous data using a line graph. Correct answer (A): \( 25 \). Student selected: \( 85 \). Underlying misconception: Adds instead of subtracts when asked to find the difference. Intervention focus: surface why the selected answer is wrong and walk through the reasoning that leads to the correct answer.

### Rubric

| Item | Score |
|---|---|
| R1 misconception engagement | 1 |
| R2 cognitive demand | 1 |
| R3 scaffolding fit | 1 |
| R4 domain accuracy | 1 |
| R5 student trajectory | 1 |
| R6 strategy fidelity | 1 |
| Quality composite (R1+R2+R3)/3 | 1.00 |

### Judge justifications

**Pass 1 (R1–R5, persona-blind):** The teacher provides completely irrelevant examples about unrelated topics (linear equations, decimal subtraction, line graphs) instead of addressing the student's misconception about treating a quadratic relationship as linear, and the student makes no progress toward understanding how to use the given equation f=3r²+3.

**Pass 2 (R6, persona-visible):** The teacher does not explain anything directly, provide worked examples, or check understanding; instead, they repeatedly paste irrelevant case examples from unrelated topics without any actual instruction.

---
