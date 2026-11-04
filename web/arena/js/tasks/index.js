// Task registry. To add a task: create ./<name>.js (see README.md) and append it here.
import tracking from "./tracking.js";
import poleBalance from "./pole_balance.js";
import whip from "./whip.js";
import pushT from "./push_t.js";
import juggling from "./juggling.js";
import catchTask from "./catch.js";
import toss from "./toss.js";

export const TASKS = [tracking, poleBalance, juggling, whip, catchTask, toss, pushT];
export const getTask = (id) => TASKS.find((t) => t.id === id) || TASKS[0];

/** fetch a task's assets: *.json -> object, anything else -> text */
export async function loadAssets(task, base = "") {
  const out = {};
  await Promise.all(Object.entries(task.assets).map(async ([key, path]) => {
    const r = await fetch(base + path);
    out[key] = path.endsWith(".json") ? await r.json() : await r.text();
  }));
  return out;
}
