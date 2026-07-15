// Shared character sheet rendering — used by both game.html (live play,
// via loadCharacterSheet()) and session_zero_index.html (lobby, via
// loadSzCharacterSheet()). Extracted 2026-07-13 so the lobby's clickable
// party cards could reuse the same renderer instead of duplicating it.

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Rarity -> CSS class, shared by inline item-links and the detail popup.
// "" (mundane, or not yet looked up) intentionally maps to no class — a
// plain link, not a miscolored one.
function rarityClass(rarity) {
  if (!rarity) return "";
  return "rarity-" + rarity.toLowerCase().replace(/\s+/g, "-");
}

// rarity is optional — pass it when already known synchronously (character
// sheet inventory has the real Item object in hand); leave it blank for a
// narration mention and enrichItemLinks() will fill the color in shortly
// after, once its background lookup resolves.
function itemLink(name, rarity) {
  const cls = ("item-link " + rarityClass(rarity)).trim();
  return `<span class="${cls}" data-item="${escapeHtml(name).replace(/"/g, "&quot;")}">${escapeHtml(name)}</span>`;
}

function abilityRow(label, score, mod) {
  const modStr = mod >= 0 ? `+${mod}` : `${mod}`;
  return `<div class="preview-score">
    <span class="score-abbr">${label}</span>
    <span class="score-val">${score}</span>
    <span class="score-mod">${modStr}</span>
  </div>`;
}

function renderCharacterSheet(d) {
  const ab = d.ability_scores || {};
  const mods = {
    strength: Math.floor((ab.strength - 10) / 2), dexterity: Math.floor((ab.dexterity - 10) / 2),
    constitution: Math.floor((ab.constitution - 10) / 2), intelligence: Math.floor((ab.intelligence - 10) / 2),
    wisdom: Math.floor((ab.wisdom - 10) / 2), charisma: Math.floor((ab.charisma - 10) / 2),
  };

  let html = `<div class="preview-identity">
    <div class="preview-name">${d.name}${d.pronouns ? ` <span class="preview-pronouns">(${d.pronouns})</span>` : ''}${d.is_player_controlled === false ? ' <span class="char-companion-tag">DM companion</span>' : ''}</div>
    <div class="preview-subtitle">${[d.race, [`${d.char_class} ${d.level}`, d.subclass].filter(Boolean).join(" / "), d.background, d.alignment].filter(Boolean).join(" · ")}</div>
  </div>`;

  html += `<div class="preview-section">
    <div class="preview-label">Combat</div>
    <div class="preview-value">
      HP: ${d.current_hp}/${d.max_hp}${d.temp_hp ? ` (+${d.temp_hp} temp)` : ""}<br>
      AC: ${d.ac} &middot; Speed: ${d.speed} ft &middot; Passive Perception: ${d.passive_perception}
      ${d.conditions && d.conditions.length ? `<br>Conditions: ${d.conditions.join(", ")}` : ""}
      ${d.exhaustion_level ? `<br>Exhaustion: ${d.exhaustion_level}` : ""}
    </div>
  </div>`;

  html += `<div class="preview-scores">
    ${abilityRow("STR", ab.strength, mods.strength)}
    ${abilityRow("DEX", ab.dexterity, mods.dexterity)}
    ${abilityRow("CON", ab.constitution, mods.constitution)}
    ${abilityRow("INT", ab.intelligence, mods.intelligence)}
    ${abilityRow("WIS", ab.wisdom, mods.wisdom)}
    ${abilityRow("CHA", ab.charisma, mods.charisma)}
  </div>`;

  if (d.saving_throw_proficiencies && d.saving_throw_proficiencies.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Saving Throws</div>
      <div class="preview-value">${d.saving_throw_proficiencies.join(", ")}</div>
    </div>`;
  }

  if (d.skill_proficiencies && d.skill_proficiencies.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Skills</div>
      <div class="preview-value">${d.skill_proficiencies.join(", ")}</div>
    </div>`;
  }

  if (d.spell_slots && Object.keys(d.spell_slots).length) {
    const slots = Object.entries(d.spell_slots)
      .filter(([, s]) => s.max > 0)
      .map(([lvl, s]) => `L${lvl}: ${s.max - s.used}/${s.max}`);
    if (slots.length) {
      html += `<div class="preview-section">
        <div class="preview-label">Spell Slots</div>
        <div class="preview-value">${slots.join(", ")}</div>
      </div>`;
    }
  }
  if (d.spells_known && d.spells_known.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Spells Known</div>
      <div class="preview-value">${d.spells_known.map(s => s.name).join(", ")}</div>
    </div>`;
  }

  if (d.attacks && d.attacks.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Attacks</div>
      <div class="preview-value">${d.attacks.map(a =>
        `${a.name} (${a.to_hit_bonus >= 0 ? "+" : ""}${a.to_hit_bonus} to hit, ${a.damage_dice} ${a.damage_type || ""})`
      ).join("<br>")}</div>
    </div>`;
  }

  if (d.inventory && d.inventory.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Inventory</div>
      <div class="preview-value">${d.inventory.map(i => itemLink(i.name, i.rarity) + (i.quantity > 1 ? ` x${i.quantity}` : "")).join(", ")}</div>
    </div>`;
  }
  if (d.currency) {
    const c = d.currency;
    const parts = [];
    if (c.pp) parts.push(`${c.pp}pp`);
    if (c.gp) parts.push(`${c.gp}gp`);
    if (c.ep) parts.push(`${c.ep}ep`);
    if (c.sp) parts.push(`${c.sp}sp`);
    if (c.cp) parts.push(`${c.cp}cp`);
    html += `<div class="preview-section">
      <div class="preview-label">Currency</div>
      <div class="preview-value">${parts.length ? parts.join(", ") : "0 gp"}</div>
    </div>`;
  }

  if (d.features && d.features.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Features</div>
      <div class="preview-value">${d.features.join(", ")}</div>
    </div>`;
  }

  if (d.appearance) {
    html += `<div class="preview-section">
      <div class="preview-label">Appearance</div>
      <div class="preview-value">${d.appearance}</div>
    </div>`;
  }

  const personality = [
    ...(d.personality_traits || []).map(t => `<em>${t}</em>`),
    ...(d.ideals || []).map(t => `Ideal: ${t}`),
    ...(d.bonds || []).map(t => `Bond: ${t}`),
    ...(d.flaws || []).map(t => `Flaw: ${t}`),
  ];
  if (personality.length) {
    html += `<div class="preview-section">
      <div class="preview-label">Personality</div>
      <div class="preview-value">${personality.join("<br>")}</div>
    </div>`;
  }

  return html;
}
