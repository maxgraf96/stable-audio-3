// SA3 plugin — WebView front end (2.3b shell).
//
// JUCE 8 injects window.__JUCE__.backend automatically when the editor is built
// with WebBrowserComponent::Options::withNativeIntegrationEnabled(true). All we
// need to invoke a backend function is to fire a "__juce__invoke" event and
// match the reply by promiseId — exactly what the JUCE-bundled JS helper does.
// Inlining it here keeps us free of the import/MIME-type complications of
// pulling in juce_gui_extra/native/javascript/index.js at this stage.

const promises = new Map();
let nextPromiseId = 0;

window.__JUCE__.backend.addEventListener("__juce__complete", ({ promiseId, result }) => {
  const handlers = promises.get(promiseId);
  if (!handlers) return;
  promises.delete(promiseId);
  handlers.resolve(result);
});

function getNativeFunction(name) {
  return (...args) => new Promise((resolve, reject) => {
    const promiseId = nextPromiseId++;
    promises.set(promiseId, { resolve, reject });
    window.__JUCE__.backend.emitEvent("__juce__invoke", {
      name,
      params: args,
      resultId: promiseId,
    });
  });
}

const statusEl = document.getElementById("status");
const getStatus = getNativeFunction("getStatus");

function statusClass(text) {
  const t = (text || "").toLowerCase();
  if (t.startsWith("load failed")) return "status status-error";
  if (t === "loaded")              return "status status-ready";
  return "status status-loading";
}

async function refresh() {
  try {
    const s = await getStatus();
    statusEl.textContent = s;
    statusEl.className   = statusClass(s);
  } catch (err) {
    statusEl.textContent = "bridge error: " + err;
    statusEl.className   = "status status-error";
  }
}

refresh();
setInterval(refresh, 200);
