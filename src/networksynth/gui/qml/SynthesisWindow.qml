import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: root
    visible: true
    width: 1280
    height: 860
    minimumWidth: 1000
    minimumHeight: 620
    title: "NetworkSynth — Synthesis"

    color: palette.window

    readonly property int labelWidth: 162
    readonly property int numberWidth: 96

    readonly property int axisWidth: 28

    readonly property int helpSize: 18

    readonly property color accent: palette.highlight
    readonly property color cardColor: palette.base
    readonly property color paneColor: palette.alternateBase
    readonly property color lineColor: palette.mid
    readonly property color railColor: Qt.darker(palette.window, 1.14)
    readonly property color plateColor: root.darkTheme ? Qt.lighter(palette.base, 1.35) : Qt.darker(palette.base, 1.05)

    readonly property bool darkTheme: palette.window.hsvValue < 0.5
    readonly property color goodColor: darkTheme ? "#5ac97f" : "#1e8e4c"
    readonly property color badColor: darkTheme ? "#f0796d" : "#c0392b"

    // font.family takes one name, and "menlo" exists on macOS only.
    readonly property string monoFamily: {
        const have = Qt.fontFamilies();
        for (const name of ["Menlo", "Consolas", "DejaVu Sans Mono", "Liberation Mono", "Courier New"])
            if (have.indexOf(name) >= 0)
                return name;
        return "monospace";
    }
    footer: Rectangle {

        color: root.cardColor
        implicitHeight: 88

        Rectangle {
            width: parent.width
            height: 1
            color: root.lineColor
        }

        ColumnLayout {
            anchors.fill: parent
            anchors.leftMargin: 18
            anchors.rightMargin: 18
            anchors.topMargin: 10
            anchors.bottomMargin: 8
            spacing: 8

            RowLayout {
                Layout.fillWidth: true
                spacing: 14

                ProgressBar {

                    visible: controller.running
                    Layout.preferredWidth: 260
                    from: 0
                    to: 100
                    value: controller.percent
                    indeterminate: controller.running && controller.percent <= 0
                }
                Label {
                    visible: controller.running
                    text: controller.percent.toFixed(0) + "%"
                    opacity: 0.7
                }
                Item {
                    Layout.fillWidth: true
                }

                Button {
                    id: runButton
                    text: controller.running ? "Cancel" : "Run " + controller.modeLabels[controller.modes.indexOf(controller.mode)]

                    enabled: controller.running || controller.canRun
                    implicitWidth: 190
                    implicitHeight: 40
                    onClicked: controller.running ? controller.cancel() : controller.run()

                    background: Rectangle {
                        radius: 6
                        color: !runButton.enabled ? root.lineColor : controller.running ? (runButton.down ? Qt.darker(root.badColor, 1.3) : root.badColor) : (runButton.down ? Qt.darker(root.accent, 1.3) : root.accent)
                    }
                    contentItem: Label {
                        text: runButton.text
                        color: runButton.enabled ? palette.highlightedText : palette.text
                        opacity: runButton.enabled ? 1.0 : 0.5
                        font.bold: true
                        horizontalAlignment: Text.AlignHCenter
                        verticalAlignment: Text.AlignVCenter
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                spacing: 8

                Label {
                    text: controller.running ? "◷" : controller.failed ? "!" : controller.canRun ? "✓" : "•"
                    opacity: controller.running || !controller.canRun ? 0.6 : 1.0
                    color: controller.failed ? root.badColor : root.goodColor
                    font.bold: true
                }
                Label {
                    Layout.fillWidth: true
                    elide: Text.ElideRight
                    text: controller.status
                    color: controller.failed ? root.badColor : palette.text
                }
                Label {
                    text: controller.runLabel
                    visible: controller.runLabel !== ""
                    font.family: root.monoFamily
                    font.pixelSize: 11
                    opacity: 0.55
                }
                Label {
                    text: controller.version
                    font.pixelSize: 11
                    opacity: 0.55
                }
                Button {
                    text: "Update"
                    visible: controller.canUpdate
                    enabled: !controller.running
                    flat: true
                    font.pixelSize: 11
                    onClicked: controller.updateTool()
                    ToolTip.text: "Fetch the newest NetworkSynth into this checkout."
                    ToolTip.visible: hovered
                }
            }
        }
    }

    property string pendingInput: ""

    MessageDialog {
        id: updateDialog
        title: "NetworkSynth"
        buttons: MessageDialog.Ok
    }

    Connections {
        target: controller
        function onUpdateDone(message, restart) {
            updateDialog.text = message;
            updateDialog.open();
        }
    }

    FileDialog {
        id: fileDialog
        onAccepted: controller.setInput(root.pendingInput, selectedFile)
    }
    FolderDialog {
        id: dirDialog
        onAccepted: controller.setInput(root.pendingInput, selectedFolder)
    }
    FolderDialog {
        id: outputDialog
        title: "Select output directory"
        onAccepted: controller.setOutputDir(selectedFolder)
    }

    function browseFor(spec) {
        root.pendingInput = spec.id;
        if (spec.kind === "dir") {
            dirDialog.title = "Select " + spec.label.toLowerCase();
            dirDialog.open();
        } else {
            fileDialog.title = "Select " + spec.label.toLowerCase();
            fileDialog.nameFilters = spec.filter ? [spec.filter, "All files (*)"] : ["All files (*)"];
            fileDialog.open();
        }
    }

    component Card: Rectangle {
        id: card
        default property alias content: body.data
        property string title: ""

        property bool collapsible: true
        property bool expanded: true

        property string note: ""

        Layout.fillWidth: true
        implicitHeight: shell.implicitHeight + 24
        color: root.cardColor
        border.color: root.lineColor
        border.width: 1
        radius: 8

        ColumnLayout {
            id: shell
            anchors.fill: parent
            anchors.margins: 12
            spacing: 10

            RowLayout {
                Layout.fillWidth: true
                spacing: 6
                visible: card.title !== ""

                Label {
                    text: (card.collapsible ? (card.expanded ? "▾  " : "▸  ") : "") + card.title.toUpperCase()
                    font.bold: true
                    font.pixelSize: 10
                    font.letterSpacing: 0.8
                    opacity: 0.55

                    TapHandler {
                        enabled: card.collapsible
                        onTapped: card.expanded = !card.expanded
                    }
                }

                InfoButton {
                    visible: card.note !== ""
                    note: card.note
                    alignRight: false
                }

                Item {
                    Layout.fillWidth: true
                }
            }
            ColumnLayout {
                id: body

                visible: card.expanded
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 8
            }
        }
    }

    component FieldEditor: RowLayout {
        id: editor
        required property var field
        required property var value
        signal edited(var newValue)
        signal sized(int index, var newValue)

        spacing: 6

        CheckBox {
            visible: editor.field.kind === "bool"

            leftPadding: 0
            checked: editor.value === true
            onToggled: editor.edited(checked)
        }

        ComboBox {
            id: choice
            visible: editor.field.kind === "choice"
            Layout.preferredWidth: 190
            model: editor.field.labels.length > 0 ? editor.field.labels : editor.field.options
            currentIndex: Math.max(0, editor.field.options.indexOf(editor.value))
            onActivated: editor.edited(editor.field.options[currentIndex])

            delegate: ItemDelegate {
                id: option
                required property var modelData
                required property int index

                width: choice.width
                text: option.modelData
                highlighted: choice.highlightedIndex === option.index

                HoverHandler {
                    id: optionHover
                }

                HoverPanel {
                    note: option.index < editor.field.option_help.length ? editor.field.option_help[option.index] : ""
                    showing: optionHover.hovered
                    x: option.width + 6
                    y: 0
                }
            }
        }

        TextField {
            visible: editor.field.kind === "text"
            Layout.preferredWidth: 210
            text: editor.field.kind === "text" ? editor.value : ""
            onEditingFinished: editor.edited(text)
        }

        SpinBox {
            visible: editor.field.kind === "integer"
            Layout.preferredWidth: 132
            editable: true

            from: editor.field.minimum != null ? editor.field.minimum : 0
            to: editor.field.maximum != null ? editor.field.maximum : 2147483647
            stepSize: editor.field.step != null ? editor.field.step : 1
            value: editor.field.kind === "integer" ? Number(editor.value) : 0
            onValueModified: editor.edited(value)
        }

        TextField {
            visible: editor.field.kind === "number"
            Layout.preferredWidth: root.numberWidth
            text: editor.field.kind === "number" ? editor.value : ""
            validator: DoubleValidator {}
            onEditingFinished: editor.edited(text)
        }

        RowLayout {
            visible: editor.field.kind === "size" || editor.field.kind === "range"
            spacing: 6

            Label {
                visible: editor.field.kind === "size"
                text: editor.field.axes[0]

                Layout.preferredWidth: root.axisWidth
                horizontalAlignment: Text.AlignRight
                opacity: 0.5
            }
            TextField {
                Layout.preferredWidth: root.numberWidth

                text: (editor.field.kind === "size" || editor.field.kind === "range") && editor.value ? editor.value[0] : ""
                placeholderText: editor.field.kind === "size" ? "auto" : ""
                validator: DoubleValidator {
                    bottom: 0
                }
                onEditingFinished: editor.sized(0, text)
            }
            Label {
                text: editor.field.kind === "range" ? "→" : "×"
                opacity: 0.5
            }
            Label {
                visible: editor.field.kind === "size"
                text: editor.field.axes[1]
                opacity: 0.5
            }
            TextField {
                Layout.preferredWidth: root.numberWidth
                text: (editor.field.kind === "size" || editor.field.kind === "range") && editor.value ? editor.value[1] : ""
                placeholderText: editor.field.kind === "size" ? "auto" : ""
                validator: DoubleValidator {
                    bottom: 0
                }
                onEditingFinished: editor.sized(1, text)
            }
        }

        Label {
            visible: editor.field.unit !== ""
            text: editor.field.unit
            opacity: 0.5
        }

        Item {
            Layout.fillWidth: true
        }
    }

    component HoverPanel: Popup {
        id: panel
        property string note: ""
        property bool showing: false
        property bool alignRight: false

        visible: panel.showing && panel.note !== ""
        closePolicy: Popup.NoAutoClose
        y: panel.parent ? panel.parent.height + 6 : 0
        x: panel.alignRight && panel.parent ? panel.parent.width - panel.width : 0
        padding: 12
        width: Math.min(380, panelText.implicitWidth + 2 * panel.padding)

        background: Rectangle {
            color: root.cardColor
            border.color: root.lineColor
            border.width: 1
            radius: 6
        }
        contentItem: Label {
            id: panelText
            text: panel.note
            wrapMode: Text.WordWrap
        }
    }

    component InfoButton: Button {
        id: info
        property string note: ""

        property bool alignRight: true

        text: "?"
        checkable: true
        implicitWidth: root.helpSize
        implicitHeight: root.helpSize
        padding: 0
        font.pixelSize: 11

        background: Rectangle {
            radius: width / 2
            color: info.checked || infoHover.hovered ? root.lineColor : "transparent"
        }

        HoverHandler {
            id: infoHover
        }

        HoverPanel {
            note: info.note
            showing: info.checked || infoHover.hovered
            alignRight: info.alignRight
        }
    }

    component FormLine: ColumnLayout {
        id: formLine
        required property var line

        readonly property string note: {
            if (formLine.line.row)
                return formLine.line.fields.filter(each => each.help !== "").map(each => each.label + " — " + each.help).join("\n");

            const one = formLine.line.fields[0];
            let text = one.help;
            if (one.kind === "choice" && one.option_help.length > 0) {
                const at = one.options.indexOf(one.current);
                if (at >= 0 && at < one.option_help.length)
                    text += (text ? "\n\n" : "") + one.option_help[at];
            }
            return text;
        }

        Layout.fillWidth: true
        spacing: 3

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Label {
                text: formLine.line.row ? formLine.line.row : formLine.line.fields[0].label

                Layout.preferredWidth: formLine.line.fields[0].kind === "size" ? root.labelWidth - root.axisWidth - 6 : root.labelWidth
                elide: Text.ElideRight
            }

            Repeater {
                model: formLine.line.row ? formLine.line.fields : []
                delegate: RowLayout {
                    id: shared
                    required property var modelData
                    spacing: 6

                    CheckBox {
                        visible: shared.modelData.kind === "bool"
                        text: shared.modelData.label === formLine.line.row
                            ? "" : shared.modelData.label
                        checked: shared.modelData.current === true
                        onToggled: controller.setValue(shared.modelData.id, checked)
                    }

                    FieldEditor {
                        visible: shared.modelData.kind !== "bool"
                        field: shared.modelData
                        value: shared.modelData.current
                        onEdited: (newValue) => controller.setValue(
                            shared.modelData.id, newValue)
                        onSized: (index, newValue) => controller.setSize(
                            shared.modelData.id, index, newValue)
                    }
                }
            }

            FieldEditor {
                visible: !formLine.line.row
                field: formLine.line.fields[0]
                value: formLine.line.fields[0].current
                onEdited: newValue => controller.setValue(formLine.line.fields[0].id, newValue)
                onSized: (index, newValue) => controller.setSize(formLine.line.fields[0].id, index, newValue)
            }

            Item {
                Layout.fillWidth: true
                visible: formLine.line.row !== ""
            }

            InfoButton {
                visible: formLine.note !== ""
                note: formLine.note
            }
        }
    }

    component PreviewPane: RowLayout {
        id: pane
        property alias image: picture.source
        property var metrics: ({})
        property string placeholder: "Run, and the result is drawn here."

        spacing: 12

        Rectangle {
            Layout.preferredWidth: 260
            Layout.preferredHeight: 260
            color: root.paneColor
            border.color: root.lineColor
            border.width: 1
            radius: 6

            Image {
                id: picture
                anchors.fill: parent
                anchors.margins: 2
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                cache: false
            }
            Label {
                anchors.centerIn: parent
                width: parent.width - 40
                horizontalAlignment: Text.AlignHCenter
                wrapMode: Text.WordWrap
                visible: picture.source == ""
                text: parent.parent.placeholder
                opacity: 0.55
            }
        }

        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 260
            color: palette.base
            border.color: root.lineColor
            border.width: 1
            radius: 6

            ScrollView {
                anchors.fill: parent
                anchors.margins: 14
                contentWidth: availableWidth
                clip: true
                // The numbers describe the figure beside them, so with nothing
                // drawn there is nothing for them to be about.
                visible: picture.source != ""

                ColumnLayout {
                    width: parent.width
                    spacing: 8

                    Label {
                        text: "NETWORK PROPERTIES"
                        visible: (pane.metrics.cells || []).length > 0
                        font.bold: true
                        font.pixelSize: 10
                        font.letterSpacing: 0.8
                        opacity: 0.55
                    }

                    GridLayout {
                        Layout.fillWidth: true
                        columns: 2
                        columnSpacing: 16
                        rowSpacing: 6

                        Repeater {
                            model: pane.metrics.cells || []

                            delegate: Label {
                                required property int index
                                required property string modelData
                                readonly property bool isValue: index % 2 === 1

                                Layout.fillWidth: isValue
                                horizontalAlignment: isValue ? Text.AlignRight : Text.AlignLeft
                                text: modelData
                                font.bold: isValue
                                opacity: isValue ? 1.0 : 0.6
                            }
                        }
                    }

                    Label {
                        Layout.topMargin: 12
                        text: "DEGREE DISTRIBUTION"
                        visible: (pane.metrics.distribution || []).length > 0
                        font.bold: true
                        font.pixelSize: 10
                        font.letterSpacing: 0.8
                        opacity: 0.55
                    }

                    Repeater {
                        model: pane.metrics.distribution || []

                        delegate: RowLayout {
                            id: share
                            required property var modelData

                            Layout.fillWidth: true
                            spacing: 10

                            Label {
                                Layout.preferredWidth: 24
                                horizontalAlignment: Text.AlignRight
                                text: share.modelData.degree
                                opacity: 0.6
                            }
                            Rectangle {
                                Layout.fillWidth: true
                                Layout.minimumWidth: 24
                                height: 4
                                radius: 2
                                color: root.lineColor

                                Rectangle {
                                    width: parent.width * share.modelData.bar
                                    height: parent.height
                                    radius: parent.radius
                                    color: root.accent
                                }
                            }
                            Label {
                                Layout.preferredWidth: 48
                                horizontalAlignment: Text.AlignRight
                                text: share.modelData.percent
                                font.bold: true
                            }
                        }
                    }
                }
            }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        Rectangle {
            Layout.preferredWidth: 196
            Layout.fillHeight: true
            color: root.railColor

            ColumnLayout {
                anchors.fill: parent
                anchors.topMargin: 18
                spacing: 2

                Label {
                    Layout.leftMargin: 18
                    Layout.bottomMargin: 12
                    text: "NetworkSynth"
                    font.bold: true
                    font.pixelSize: 15
                }
                Label {
                    Layout.leftMargin: 18
                    Layout.bottomMargin: 4
                    text: "MODE"
                    opacity: 0.5
                    font.pixelSize: 10
                    font.letterSpacing: 0.8
                    font.bold: true
                }

                Repeater {
                    model: controller.modeLabels

                    delegate: ItemDelegate {
                        id: modeItem
                        required property int index
                        required property var modelData
                        readonly property bool current: controller.modes[index] === controller.mode

                        Layout.fillWidth: true
                        implicitHeight: 38
                        enabled: !controller.running
                        onClicked: controller.selectMode(modeItem.index)

                        background: Rectangle {
                            color: modeItem.current ? root.accent : modeItem.hovered ? Qt.darker(root.railColor, 1.12) : "transparent"
                        }
                        contentItem: Label {
                            leftPadding: 18
                            verticalAlignment: Text.AlignVCenter
                            text: modeItem.modelData
                            color: modeItem.current ? palette.highlightedText : palette.text
                            font.bold: modeItem.current
                            opacity: modeItem.enabled ? (modeItem.current ? 1.0 : 0.8) : 0.4
                        }
                    }
                }

                Item {
                    Layout.fillHeight: true
                }
            }
        }

        ColumnLayout {
            Layout.fillWidth: true
            Layout.fillHeight: true
            Layout.margins: 18
            spacing: 2

            Label {
                text: controller.modeLabels[controller.modes.indexOf(controller.mode)]
                font.pixelSize: 22
                font.bold: true
            }
            Label {
                Layout.fillWidth: true
                Layout.bottomMargin: 10
                text: controller.modeBlurb
                opacity: 0.65
                elide: Text.ElideRight
            }

            SplitView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                orientation: Qt.Horizontal

                ScrollView {
                    SplitView.preferredWidth: 500
                    SplitView.minimumWidth: 420
                    contentWidth: availableWidth

                    ColumnLayout {
                        width: parent.width
                        spacing: 12

                        Card {
                            title: "Input"

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: controller.hasInputChoice

                                Label {
                                    text: "Input"
                                    Layout.preferredWidth: root.labelWidth
                                }
                                Repeater {
                                    model: controller.inputScopes
                                    delegate: RadioButton {
                                        required property int index
                                        required property var modelData
                                        text: modelData
                                        checked: index === controller.inputScope
                                        enabled: !controller.running
                                        onClicked: controller.selectInputScope(index)
                                    }
                                }
                                Item {
                                    Layout.fillWidth: true
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: controller.inputFormats.length > 1

                                Label {
                                    text: "Format"
                                    Layout.preferredWidth: root.labelWidth
                                }
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: controller.inputFormats
                                    currentIndex: controller.inputFormat
                                    onActivated: controller.selectInputFormat(currentIndex)

                                    enabled: !controller.running
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: controller.hasHandover

                                Item {
                                    Layout.preferredWidth: root.labelWidth
                                }
                                Button {
                                    text: "Use the StructuralGT network"
                                    enabled: !controller.running
                                    onClicked: controller.reloadHandover()
                                    ToolTip.text: "Put the network StructuralGT sent back in the fields above."
                                    ToolTip.visible: hovered
                                }
                                Item {
                                    Layout.fillWidth: true
                                }
                            }

                            Repeater {
                                model: controller.inputs

                                delegate: RowLayout {
                                    id: inputRow
                                    required property var modelData
                                    Layout.fillWidth: true
                                    spacing: 10

                                    Label {
                                        text: inputRow.modelData.label
                                        Layout.preferredWidth: root.labelWidth
                                    }
                                    TextField {
                                        id: inputField
                                        Layout.fillWidth: true
                                        text: inputRow.modelData.value
                                        placeholderText: inputRow.modelData.placeholder
                                        onEditingFinished: controller.setInput(inputRow.modelData.id, text)

                                        MouseArea {
                                            anchors.fill: parent
                                            enabled: inputField.text === ""
                                            onClicked: root.browseFor(inputRow.modelData)
                                        }
                                    }

                                    Label {
                                        Layout.preferredWidth: 14
                                        horizontalAlignment: Text.AlignHCenter
                                        font.bold: true
                                        text: inputRow.modelData.state === "ok" ? "✓" : inputRow.modelData.state === "missing" ? "!" : ""
                                        color: inputRow.modelData.state === "ok" ? root.goodColor : root.badColor
                                        HoverHandler {
                                            id: stateHover
                                        }
                                        HoverPanel {
                                            note: inputRow.modelData.state === "ok" ? "Found." : "Needed, and not found yet."
                                            showing: stateHover.hovered && inputRow.modelData.state !== "blank"
                                        }
                                    }
                                    Button {
                                        text: "Browse…"
                                        onClicked: root.browseFor(inputRow.modelData)
                                    }

                                    InfoButton {
                                        visible: inputRow.modelData.help !== ""
                                        note: inputRow.modelData.help
                                    }
                                }
                            }
                        }

                        Card {
                            title: "Align"

                            expanded: false
                            visible: controller.alignLines.length > 0 && controller.canPreview

                            Repeater {
                                model: controller.alignLines
                                delegate: FormLine {
                                    required property var modelData
                                    line: modelData
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8
                                visible: controller.canFitFrame

                                Label {
                                    text: "Fit to StructuralGT"
                                    Layout.preferredWidth: root.labelWidth
                                }
                                ComboBox {
                                    Layout.preferredWidth: 130
                                    model: controller.structuralgtSides
                                    currentIndex: controller.structuralgtSide
                                    enabled: !controller.running
                                    displayText: currentText + " px"
                                    onActivated: controller.selectStructuralgtSide(currentIndex)
                                }
                                Button {
                                    text: "Fit"
                                    enabled: !controller.running
                                    onClicked: controller.fitFrame()
                                    ToolTip.text: "Set the window to the copy StructuralGT traced the network from."
                                    ToolTip.visible: hovered
                                }
                                Item {
                                    Layout.fillWidth: true
                                }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 8

                                Item {
                                    Layout.preferredWidth: root.labelWidth
                                }
                                Button {
                                    text: "Save as edited"
                                    enabled: controller.canRun
                                    onClicked: controller.saveEdited()
                                    HoverPanel {
                                        note: "Write the network as it is drawn " + "now into an 'edited' folder beside " + "the source, and read that from here on."
                                        showing: parent.hovered
                                    }
                                }
                                Item {
                                    Layout.fillWidth: true
                                }
                            }
                        }

                        Repeater {
                            model: controller.configSections

                            delegate: Card {
                                id: sectionCard
                                required property var modelData
                                title: sectionCard.modelData.name
                                note: sectionCard.modelData.note

                                Repeater {
                                    model: sectionCard.modelData.lines
                                    delegate: FormLine {
                                        required property var modelData
                                        line: modelData
                                    }
                                }
                            }
                        }

                        Item {
                            Layout.fillHeight: true
                        }
                    }
                }

                ScrollView {
                    SplitView.fillWidth: true
                    SplitView.minimumWidth: 470
                    contentWidth: availableWidth

                    ColumnLayout {
                        width: parent.width
                        spacing: 12

                        Card {
                            title: "Output"
                            Layout.leftMargin: 14

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                Label {
                                    text: "Folder"
                                    Layout.preferredWidth: root.labelWidth
                                }
                                TextField {
                                    id: outputField
                                    Layout.fillWidth: true
                                    text: controller.outputDir
                                    onEditingFinished: controller.setOutputDir(text)

                                    MouseArea {
                                        anchors.fill: parent
                                        enabled: outputField.text === ""
                                        onClicked: outputDialog.open()
                                    }
                                }
                                Button {
                                    text: "Browse…"
                                    onClicked: outputDialog.open()
                                }

                                InfoButton {
                                    note: "Each run makes its own timestamped " + "folder in here, named for the mode and " + "the run. \"latest_result\" is kept " + "pointing at the newest."
                                }
                            }

                            Repeater {
                                model: controller.outputLines
                                delegate: FormLine {
                                    required property var modelData
                                    line: modelData
                                }
                            }

                            ColumnLayout {
                                Layout.fillWidth: true
                                Layout.topMargin: 4
                                spacing: 8

                                RowLayout {
                                    Layout.fillWidth: false
                                    spacing: 14

                                    Rectangle {
                                        Layout.preferredWidth: 16
                                        height: 1
                                        color: root.lineColor
                                    }

                                    Label {
                                        text: "OUTPUT FORMATS"
                                        font.pixelSize: 10
                                        font.bold: true
                                        font.letterSpacing: 0.8
                                        opacity: 0.55
                                    }

                                    Rectangle {
                                        Layout.fillWidth: true
                                        height: 1
                                        color: root.lineColor
                                    }

                                    Label {
                                        text: "Select none"
                                        font.pixelSize: 11
                                        color: noneHover.hovered ? root.accent : palette.text
                                        opacity: noneHover.hovered ? 1.0 : 0.55

                                        HoverHandler {
                                            id: noneHover
                                            cursorShape: Qt.PointingHandCursor
                                        }
                                        TapHandler {
                                            onTapped: controller.clearOutputs()
                                        }
                                        HoverPanel {
                                            note: "Write no networks and no plots. " + "A run still leaves its report " + "and the batch a preview reads."
                                            showing: noneHover.hovered
                                            alignRight: true
                                        }
                                    }

                                    Label {
                                        text: "Defaults"
                                        font.pixelSize: 11
                                        color: defaultsHover.hovered ? root.accent : palette.text
                                        opacity: defaultsHover.hovered ? 1.0 : 0.55

                                        HoverHandler {
                                            id: defaultsHover
                                            cursorShape: Qt.PointingHandCursor
                                        }
                                        TapHandler {
                                            onTapped: controller.resetOutputs()
                                        }
                                        HoverPanel {
                                            note: "Networks as .csv, plots as .webp."
                                            showing: defaultsHover.hovered
                                            alignRight: true
                                        }
                                    }
                                }

                                ColumnLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Repeater {
                                        model: controller.formatLines
                                        delegate: FormLine {
                                            required property var modelData
                                            line: modelData
                                        }
                                    }
                                }

                                Rectangle {
                                    Layout.fillWidth: true
                                    height: 1
                                    color: root.lineColor
                                    Layout.bottomMargin: 4
                                }
                            }
                        }

                        Card {
                            title: "Preview"
                            Layout.leftMargin: 14
                            visible: controller.canPreview

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10

                                RowLayout {
                                    id: previewTabs
                                    property int currentIndex: 0
                                    onCurrentIndexChanged: controller.selectPreviewTab(currentIndex)
                                    spacing: 4

                                    Connections {
                                        target: controller
                                        function onChanged() {
                                            if (!controller.offersSyntheticPreview && previewTabs.currentIndex !== 0)
                                                previewTabs.currentIndex = 0;
                                        }
                                    }

                                    Rectangle {
                                        Layout.preferredWidth: 112
                                        Layout.preferredHeight: 26
                                        radius: 6
                                        color: previewTabs.currentIndex === 0 ? root.paneColor : "transparent"

                                        Label {
                                            anchors.centerIn: parent
                                            text: "Original"
                                            opacity: previewTabs.currentIndex === 0 ? 1.0 : 0.65
                                        }
                                        TapHandler {
                                            onTapped: previewTabs.currentIndex = 0
                                        }
                                    }

                                    Rectangle {
                                        Layout.preferredWidth: 112
                                        Layout.preferredHeight: 26
                                        radius: 6
                                        visible: controller.offersSyntheticPreview
                                        enabled: controller.canPreviewSynthetic
                                        opacity: enabled ? 1.0 : 0.4
                                        color: previewTabs.currentIndex === 1 ? root.paneColor : "transparent"

                                        Label {
                                            anchors.centerIn: parent
                                            text: "Synthetic"
                                            opacity: previewTabs.currentIndex === 1 ? 1.0 : 0.65
                                        }
                                        TapHandler {
                                            onTapped: previewTabs.currentIndex = 1
                                        }
                                    }
                                }

                                Button {
                                    text: "Background"
                                    checkable: true
                                    checked: controller.showBackground
                                    visible: previewTabs.currentIndex === 0 && controller.hasBackground
                                    onToggled: controller.setShowBackground(checked)
                                }
                                Button {
                                    text: "Network"
                                    checkable: true
                                    checked: controller.showNetwork
                                    visible: previewTabs.currentIndex === 0
                                    onToggled: controller.setShowNetwork(checked)
                                }
                                Item {
                                    Layout.fillWidth: true
                                }
                                Label {
                                    text: controller.previewStatus
                                    color: controller.previewFailed ? root.badColor : palette.text
                                    opacity: controller.previewFailed ? 1.0 : 0.65
                                    elide: Text.ElideRight
                                }
                            }

                            Label {
                                Layout.fillWidth: true
                                wrapMode: Text.WordWrap
                                opacity: 0.65
                                text: previewTabs.currentIndex === 0 ? controller.originalNote : controller.syntheticNote
                                visible: text !== ""
                            }

                            StackLayout {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 264
                                currentIndex: previewTabs.currentIndex

                                PreviewPane {
                                    image: controller.originalImage
                                    metrics: controller.originalInfo
                                    placeholder: controller.previewStatus !== "" ? controller.previewStatus : "Choose an edge list and positions, and the input network is drawn here."
                                }
                                PreviewPane {
                                    image: controller.syntheticImage
                                    metrics: controller.syntheticInfo
                                    placeholder: controller.previewStatus !== "" ? controller.previewStatus : "Run, and the first network of the output is drawn here."
                                }
                            }
                        }

                        Card {
                            title: "Log"
                            Layout.leftMargin: 14

                            ScrollView {
                                Layout.fillWidth: true
                                Layout.preferredHeight: 220
                                TextArea {
                                    id: logArea
                                    readOnly: true
                                    wrapMode: TextArea.NoWrap
                                    font.family: root.monoFamily
                                    font.pixelSize: 11
                                    text: controller.logText
                                    background: null
                                    onTextChanged: cursorPosition = length
                                }
                            }
                        }
                    }
                }
            }
        }
    }

}
