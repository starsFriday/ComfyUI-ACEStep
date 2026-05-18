import { app } from "../../scripts/app.js";

const ACE_NODE_COLOR = "#2B6E7F";
const ACE_NODE_BGCOLOR = "#18323D";

const ACE_NODE_NAMES = new Set([
  "ACEStep15XLPromptLyrics",
  "ACEStep15XLTextEncode",
  "ACEStep15XLTTSLikePrompt",
  "ACEStep15XLEmptyLatentAudio",
  "ACEStep15XLAudioToLatent",
  "ACEStep15XLReferenceAudio",
  "ACEStep15XLReferenceLatent",
  "ACEStep15XLExtendAudio",
  "ACEStep15XLExtendLatent",
  "ACEStep15XLRepaintAudio",
  "ACEStep15XLRepaintLatent",
  "ACEStep15XLEditAudio",
  "ACEStep15XLEditLatent",
]);

function isAceNode(nodeData) {
  return ACE_NODE_NAMES.has(nodeData?.name) || String(nodeData?.name || "").startsWith("ACEStep15XL");
}

function applyAceColor(node) {
  node.color = ACE_NODE_COLOR;
  node.bgcolor = ACE_NODE_BGCOLOR;
}

app.registerExtension({
  name: "ComfyUI.ACEStep.Appearance",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!isAceNode(nodeData)) {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);
      applyAceColor(this);
      return result;
    };
  },
});
