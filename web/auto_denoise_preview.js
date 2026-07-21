import { app } from "../../scripts/app.js";
import {
    addTextPreviewWidgets,
    updateTextPreviewWidgets,
} from "../core/textPreviewWidgets.js";


const NODE_ID = "UltimateFaceFixAutoDenoise";


app.registerExtension({
    name: "ultimate-face-fix.auto-denoise-preview",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_ID) {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            addTextPreviewWidgets(this);

            const previewMode = this.widgets?.find(
                (widget) => widget.name === "preview_mode",
            );
            if (previewMode) {
                previewMode.options.hidden = true;
            }

            updateTextPreviewWidgets(this, { text: ["Waiting for execution"] });
            return result;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            updateTextPreviewWidgets(this, message);
        };
    },
});
