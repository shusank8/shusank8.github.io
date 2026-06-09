document.addEventListener("DOMContentLoaded", () => {
  const content = document.querySelector(".page__content");

  if (!content) return;

  const headings = content.querySelectorAll("h2");

  headings.forEach((heading) => {
    const next = heading.nextElementSibling;

    if (!next || next.tagName !== "UL") return;

    const dayBox = document.createElement("details");
    dayBox.className = "todobox";
    dayBox.open = false;

    const daySummary = document.createElement("summary");
    daySummary.textContent = heading.textContent;

    dayBox.appendChild(daySummary);

    next.querySelectorAll(":scope > li").forEach((li) => {
      const note = li.querySelector("blockquote");

      // Check if markdown rendered an actual checkbox
      const checkbox = li.querySelector('input[type="checkbox"]');
      let completed = checkbox?.checked || false;

      // Clone so we can safely remove note/checkbox
      const clone = li.cloneNode(true);

      clone.querySelectorAll("blockquote").forEach((el) => el.remove());

      clone
        .querySelectorAll('input[type="checkbox"]')
        .forEach((el) => el.remove());

      let taskText = clone.textContent.trim();

      // Fallback if checkbox wasn't rendered by markdown
      if (!checkbox) {
        if (/^\[x\]\s*/i.test(taskText)) {
          completed = true;
          taskText = taskText.replace(/^\[x\]\s*/i, "");
        } else if (/^\[\s\]\s*/i.test(taskText)) {
          taskText = taskText.replace(/^\[\s\]\s*/i, "");
        }
      }

      const todo = document.createElement("details");
      todo.className = "todoitem";

      if (completed) {
        todo.classList.add("completed");
      }

      const summary = document.createElement("summary");
      summary.textContent = (completed ? "☑ " : "☐ ") + taskText;

      todo.appendChild(summary);

      if (note) {
        const noteDiv = document.createElement("div");
        noteDiv.className = "todonote";
        noteDiv.innerHTML = note.innerHTML;

        todo.appendChild(noteDiv);
      }

      if (note) {
        summary.classList.add("has-note");
      }

      dayBox.appendChild(todo);
    });

    heading.replaceWith(dayBox);
    next.remove();
  });
});
