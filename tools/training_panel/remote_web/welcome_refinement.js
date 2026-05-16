(() => {
  const TRAIN_LABEL = "Go To Training Page";

  function activeView() {
    return document.querySelector(".nav-tabs button.active[data-view]")?.dataset.view || "";
  }

  function refineWelcomeActions() {
    const view = activeView();
    document.querySelectorAll('.welcome-actions button[data-action="view"][data-view="train"]').forEach((button) => {
      button.textContent = TRAIN_LABEL;
      button.hidden = view === "train";
    });
    document.querySelectorAll('.welcome-actions button[data-action="view"][data-view="history"]').forEach((button) => {
      button.hidden = view === "history";
    });
  }

  const observer = new MutationObserver(refineWelcomeActions);
  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ["class"],
  });

  document.addEventListener("click", (event) => {
    if (event.target.closest('[data-action="view"]')) {
      window.setTimeout(refineWelcomeActions, 0);
    }
  }, true);

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", refineWelcomeActions);
  } else {
    refineWelcomeActions();
  }
})();
