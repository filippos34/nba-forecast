// Build-time loader: the pipeline writes public/data/*.json (src/site_export.py); pages read them here.
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const DIR = join(process.cwd(), "public", "data");

export function load<T = any>(name: string, fallback: T): T {
  const p = join(DIR, name);
  if (!existsSync(p)) return fallback;
  return JSON.parse(readFileSync(p, "utf-8")) as T;
}

export const pct = (x: number | null | undefined, d = 0) =>
  x === null || x === undefined ? "—" : `${(x * 100).toFixed(d)}%`;
export const num = (x: number | null | undefined, d = 4) =>
  x === null || x === undefined ? "—" : x.toFixed(d);
export const signed = (x: number | null | undefined, d = 4) =>
  x === null || x === undefined ? "—" : (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);
export const etTime = (iso: string) =>
  new Date(iso).toLocaleTimeString("en-US", { timeZone: "America/New_York", hour: "numeric", minute: "2-digit" }) + " ET";
export const etDate = (iso: string) =>
  new Date(iso + "T12:00:00Z").toLocaleDateString("en-US", { weekday: "long", month: "long", day: "numeric", year: "numeric" });

export const TEAM: Record<string, string> = {
  ATL: "Atlanta Hawks", BOS: "Boston Celtics", BKN: "Brooklyn Nets", CHA: "Charlotte Hornets", CHI: "Chicago Bulls",
  CLE: "Cleveland Cavaliers", DAL: "Dallas Mavericks", DEN: "Denver Nuggets", DET: "Detroit Pistons", GS: "Golden State Warriors",
  HOU: "Houston Rockets", IND: "Indiana Pacers", LAC: "LA Clippers", LAL: "Los Angeles Lakers", MEM: "Memphis Grizzlies",
  MIA: "Miami Heat", MIL: "Milwaukee Bucks", MIN: "Minnesota Timberwolves", NO: "New Orleans Pelicans", NY: "New York Knicks",
  OKC: "Oklahoma City Thunder", ORL: "Orlando Magic", PHI: "Philadelphia 76ers", PHX: "Phoenix Suns", POR: "Portland Trail Blazers",
  SAC: "Sacramento Kings", SA: "San Antonio Spurs", TOR: "Toronto Raptors", UTAH: "Utah Jazz", WSH: "Washington Wizards",
};
