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
    // Every surface below comes from the system palette, so light and dark
    // both follow the desktop.  A colour written here would be frozen at one
    // of them, and mixing the two is what makes a window unreadable.
    color: palette.window

    // `controller` is exposed from Python as a context property.
    //
    // A rail of modes on the left, the chosen mode's page beside it, and that
    // page split into what a run is given and what it produces.  Run and state
    // live in the footer, so the button pressed every time is never scrolled
    // off and the last run stays named.

    readonly property int labelWidth: 162
    readonly property int numberWidth: 96
    //: The gutter an axis letter sits in, taken out of the label beside it.
    readonly property int axisWidth: 28

    readonly property color accent: palette.highlight
    readonly property color cardColor: palette.base
    readonly property color paneColor: palette.alternateBase
    readonly property color lineColor: palette.mid
    readonly property color railColor: Qt.darker(palette.window, 1.14)
    readonly property color plateColor: root.darkTheme
        ? Qt.lighter(palette.base, 1.35) : Qt.darker(palette.base, 1.05)

    //: Which way round the desktop is, for the two colours that carry a
    //: meaning rather than a role: a success green and a failure red have no
    //: palette entry, and one pair cannot be legible on both.
    readonly property bool darkTheme: palette.window.hsvValue < 0.5
    readonly property color goodColor: darkTheme ? "#5ac97f" : "#1e8e4c"
    readonly property color badColor: darkTheme ? "#f0796d" : "#c0392b"

    footer: Rectangle {
        // root.cardColor, not `palette.base` read here: a bare Rectangle is not
        // a Control, and Qt resolves its palette from a different source than
        // the one the controls use.  Reading both is how a window ends up
        // half-themed, which is the whole bug this file used to have.
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
                    // Only while there is progress to show: an empty trough
                    // beside the button reads as something broken.
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
                Item { Layout.fillWidth: true }

                Button {
                    id: runButton
                    text: controller.running
                        ? "Cancel"
                        : "Run " + controller.modeLabels[
                            controller.modes.indexOf(controller.mode)]
                    // Dead rather than accepting a click it would only refuse.
                    // What is missing is named in the status line below.
                    enabled: controller.running || controller.canRun
                    implicitWidth: 190
                    implicitHeight: 40
                    onClicked: controller.running ? controller.cancel() : controller.run()

                    background: Rectangle {
                        radius: 6
                        color: !runButton.enabled ? root.lineColor
                             : controller.running
                                 ? (runButton.down
                                     ? Qt.darker(root.badColor, 1.3) : root.badColor)
                                 : (runButton.down
                                     ? Qt.darker(root.accent, 1.3) : root.accent)
                    }
                    contentItem: Label {
                        text: runButton.text
                        color: runButton.enabled
                            ? palette.highlightedText : palette.text
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
                    text: controller.running ? "◷"
                        : controller.failed ? "!"
                        : controller.canRun ? "✓" : "•"
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
                    font.family: "monospace"
                    font.pixelSize: 11
                    opacity: 0.55
                }
            }
        }
    }

    // One dialog pair reused by every input row: which one opens, and which
    // input receives the answer, is decided when Browse is clicked.
    property string pendingInput: ""

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
        root.pendingInput = spec.id
        if (spec.kind === "dir") {
            dirDialog.title = "Select " + spec.label.toLowerCase()
            dirDialog.open()
        } else {
            fileDialog.title = "Select " + spec.label.toLowerCase()
            fileDialog.nameFilters = spec.filter
                ? [spec.filter, "All files (*)"] : ["All files (*)"]
            fileDialog.open()
        }
    }

    // A titled panel.  Children go straight inside it.
    component Card: Rectangle {
        id: card
        default property alias content: body.data
        property string title: ""
        //: A card that can be folded away, for one whose rows are only wanted
        //: now and then.  The heading keeps saying it is there.
        property bool collapsible: false
        property bool expanded: true
        //: What this section is about, behind the mark beside its heading.
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
                    text: (card.collapsible ? (card.expanded ? "▾  " : "▸  ") : "")
                        + card.title.toUpperCase()
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

                Item { Layout.fillWidth: true }
            }
            ColumnLayout {
                id: body
                // Layouts skip an invisible child, so a folded card shrinks to
                // its heading rather than leaving the gap behind.
                visible: card.expanded
                Layout.fillWidth: true
                Layout.fillHeight: true
                spacing: 8
            }
        }
    }

    // One editor per field kind, as siblings gated on `visible` rather than a
    // Loader with inline Components: those get a context where neither
    // `controller` nor `modelData` resolves.  QtQuick.Layouts skips invisible
    // items, so only the relevant one takes space.
    component FieldEditor: RowLayout {
        id: editor
        required property var field
        required property var value
        signal edited(var newValue)
        signal sized(int index, var newValue)

        spacing: 6

        CheckBox {
            visible: editor.field.kind === "bool"
            // No indent of its own: the indicator lines up with the editors
            // above and below it.
            leftPadding: 0
            checked: editor.value === true
            onToggled: editor.edited(checked)
        }

        // Named for the reader, valued for the run: a gate is registered
        // under a key and a moment range is a boolean, and neither is
        // anything to put in front of someone.
        ComboBox {
            visible: editor.field.kind === "choice"
            Layout.preferredWidth: 190
            model: editor.field.labels.length > 0
                ? editor.field.labels : editor.field.options
            currentIndex: Math.max(0, editor.field.options.indexOf(editor.value))
            onActivated: editor.edited(editor.field.options[currentIndex])
        }

        TextField {
            visible: editor.field.kind === "text"
            Layout.preferredWidth: 210
            text: editor.field.kind === "text" ? editor.value : ""
            onEditingFinished: editor.edited(text)
        }

        // Whole numbers get a spin box: the bounds and the step are already
        // declared on the field, so the control can hold to them rather than
        // the form complaining afterwards.
        SpinBox {
            visible: editor.field.kind === "integer"
            Layout.preferredWidth: 132
            editable: true
            // Loose `!=`, because an unset bound arrives from Python as
            // undefined rather than null, and a strict test lets it through.
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

            // Which box is which, rather than leaving it to the order.  A
            // range is a start and an end, so it gets no letters.
            Label {
                visible: editor.field.kind === "size"
                text: editor.field.axes[0]
                // A slot of its own, so the box after it starts where every
                // other editor does rather than one word further right.
                Layout.preferredWidth: root.axisWidth
                horizontalAlignment: Text.AlignRight
                opacity: 0.5
            }
            TextField {
                Layout.preferredWidth: root.numberWidth
                // Bindings of invisible siblings still evaluate, so guard the
                // indexing: on a scalar field this would be undefined.
                text: editor.field.kind === "size" || editor.field.kind === "range"
                    ? editor.value[0] : ""
                validator: DoubleValidator { bottom: 0 }
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
                text: editor.field.kind === "size" || editor.field.kind === "range"
                    ? editor.value[1] : ""
                validator: DoubleValidator { bottom: 0 }
                onEditingFinished: editor.sized(1, text)
            }
        }

        // What the number is counted in.  Positions are image pixels, which
        // is why a frame is too — the two have to be the same space for the
        // network to land on its background.
        Label {
            visible: editor.field.unit !== ""
            text: editor.field.unit
            opacity: 0.5
        }

        Item { Layout.fillWidth: true }
    }

    // Every explanation in the window is drawn here: ours rather than the
    // style's, because an attached ToolTip is drawn by the platform and does
    // not look the same on every one of them.
    component HoverPanel: Popup {
        id: panel
        property string note: ""
        property bool showing: false
        property bool alignRight: false

        visible: panel.showing && panel.note !== ""
        closePolicy: Popup.NoAutoClose
        y: panel.parent ? panel.parent.height + 6 : 0
        x: panel.alignRight && panel.parent
            ? panel.parent.width - panel.width : 0
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

    // What something does, behind a button rather than beside it: an
    // explanation that runs to a paragraph does not belong in the flow of rows
    // a form is scanned down.  Shown while the pointer rests on the button,
    // and held up by a click for anything worth reading twice.
    component InfoButton: Button {
        id: info
        property string note: ""
        //: Panels hang left from a button in the help column at the right, and
        //: right from one beside a heading, so neither runs off the card.
        property bool alignRight: true

        text: "?"
        checkable: true
        implicitWidth: 34

        HoverHandler { id: infoHover }

        HoverPanel {
            note: info.note
            showing: info.checked || infoHover.hovered
            alignRight: info.alignRight
        }
    }

    // One form row: a single field, or a set of switches sharing a label —
    // and, for a choice, a button holding what the current pick does.
    component FormLine: ColumnLayout {
        id: formLine
        required property var line

        // Everything this row has to say, as one note: what the field is for,
        // and under a choice what the current pick does.  A row of switches
        // speaks for each of them in turn, since the label they share cannot.
        readonly property string note: {
            if (formLine.line.row)
                return formLine.line.fields
                    .filter(each => each.help !== "")
                    .map(each => each.label + " — " + each.help)
                    .join("\n")

            const one = formLine.line.fields[0]
            let text = one.help
            if (one.kind === "choice" && one.option_help.length > 0) {
                const at = one.options.indexOf(one.current)
                if (at >= 0 && at < one.option_help.length)
                    text += (text ? "\n\n" : "") + one.option_help[at]
            }
            return text
        }

        Layout.fillWidth: true
        spacing: 3

        RowLayout {
            Layout.fillWidth: true
            spacing: 10

            Label {
                text: formLine.line.row ? formLine.line.row : formLine.line.fields[0].label
                // Narrower on a row whose editor is prefixed by an axis
                // letter, by exactly what that letter takes: the boxes down
                // the page all start at the same x either way.
                Layout.preferredWidth: formLine.line.fields[0].kind === "size"
                    ? root.labelWidth - root.axisWidth - 6 : root.labelWidth
                elide: Text.ElideRight
            }

            // A set: every format on one line, each plainly a switch to turn on or
            // off rather than a badge stating what is already chosen.
            Repeater {
                model: formLine.line.row ? formLine.line.fields : []
                delegate: CheckBox {
                    required property var modelData
                    text: modelData.label
                    checked: modelData.current === true
                    onToggled: controller.setValue(modelData.id, checked)
                }
            }

            FieldEditor {
                visible: !formLine.line.row
                field: formLine.line.fields[0]
                value: formLine.line.fields[0].current
                onEdited: (newValue) => controller.setValue(
                    formLine.line.fields[0].id, newValue)
                onSized: (index, newValue) => controller.setSize(
                    formLine.line.fields[0].id, index, newValue)
            }

            // Only where there is no editor to take up the slack: a switch row
            // keeps its switches to the left, and every other row keeps its
            // help button out at the edge, in the column the input rows put
            // theirs in.
            Item { Layout.fillWidth: true; visible: formLine.line.row !== "" }

            InfoButton {
                visible: formLine.note !== ""
                note: formLine.note
            }
        }
    }

    component PreviewPane: RowLayout {
        property alias image: picture.source
        property alias info: details.text
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
            color: root.paneColor
            border.color: root.lineColor
            border.width: 1
            radius: 6

            ScrollView {
                anchors.fill: parent
                anchors.margins: 8
                TextArea {
                    id: details
                    readOnly: true
                    wrapMode: TextArea.NoWrap
                    font.family: "monospace"
                    font.pixelSize: 11
                    background: null
                }
            }
        }
    }

    RowLayout {
        anchors.fill: parent
        spacing: 0

        // ---- the modes, as a rail ----
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
                        readonly property bool current:
                            controller.modes[index] === controller.mode

                        Layout.fillWidth: true
                        implicitHeight: 38
                        enabled: !controller.running
                        onClicked: controller.selectMode(modeItem.index)

                        background: Rectangle {
                            color: modeItem.current ? root.accent
                                 : modeItem.hovered
                                     ? Qt.darker(root.railColor, 1.12) : "transparent"
                        }
                        contentItem: Label {
                            leftPadding: 18
                            verticalAlignment: Text.AlignVCenter
                            text: modeItem.modelData
                            color: modeItem.current
                                ? palette.highlightedText : palette.text
                            font.bold: modeItem.current
                            opacity: modeItem.enabled
                                ? (modeItem.current ? 1.0 : 0.8) : 0.4
                        }
                    }
                }

                Item { Layout.fillHeight: true }
            }
        }

        // ---- the chosen mode's page ----
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

                // what the run is given
                ScrollView {
                    SplitView.preferredWidth: 500
                    SplitView.minimumWidth: 420
                    contentWidth: availableWidth

                    ColumnLayout {
                        width: parent.width
                        spacing: 12

                        Card {
                            title: "Input"

                            // How many, and what of, as two questions rather
                            // than one list of every combination.  Only where
                            // the mode offers a choice: analysis reads a pair
                            // of results folders and nothing else.
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
                                Item { Layout.fillWidth: true }
                            }

                            RowLayout {
                                Layout.fillWidth: true
                                spacing: 10
                                visible: controller.hasInputChoice

                                Label {
                                    text: "Format"
                                    Layout.preferredWidth: root.labelWidth
                                }
                                ComboBox {
                                    Layout.fillWidth: true
                                    model: controller.inputFormats
                                    currentIndex: controller.inputFormat
                                    onActivated: controller.selectInputFormat(currentIndex)
                                    // A folder is only read one way, so there
                                    // is nothing to choose — shown rather than
                                    // hidden, so it still says what it will be.
                                    enabled: !controller.running
                                        && controller.inputFormats.length > 1
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
                                        onEditingFinished:
                                            controller.setInput(inputRow.modelData.id, text)

                                        // An empty box asks for the file when
                                        // clicked; once it holds a path it goes
                                        // back to being editable, so a typed or
                                        // pasted one is still possible.
                                        MouseArea {
                                            anchors.fill: parent
                                            enabled: inputField.text === ""
                                            onClicked: root.browseFor(inputRow.modelData)
                                        }
                                    }
                                    // Whether this file is settled, beside the
                                    // field rather than only in the run's
                                    // refusal at the foot of the window.
                                    Label {
                                        Layout.preferredWidth: 14
                                        horizontalAlignment: Text.AlignHCenter
                                        font.bold: true
                                        text: inputRow.modelData.state === "ok" ? "✓"
                                            : inputRow.modelData.state === "missing"
                                                ? "!" : ""
                                        color: inputRow.modelData.state === "ok"
                                            ? root.goodColor : root.badColor
                                        HoverHandler { id: stateHover }
                                        HoverPanel {
                                            note: inputRow.modelData.state === "ok"
                                                ? "Found."
                                                : "Needed, and not found yet."
                                            showing: stateHover.hovered
                                                && inputRow.modelData.state !== "blank"
                                        }
                                    }
                                    Button {
                                        text: "Browse…"
                                        onClicked: root.browseFor(inputRow.modelData)
                                    }
                                    // What this input's format has to contain.
                                    // Each shape reads a different one, and the
                                    // wrong file fails deep inside a run rather
                                    // than here.
                                    InfoButton {
                                        visible: inputRow.modelData.help !== ""
                                        note: inputRow.modelData.help
                                    }
                                }
                            }
                        }

                        Card {
                            title: "Align"
                            // Folded: squaring the input up with its image is
                            // something you do once, if ever.
                            collapsible: true
                            expanded: false
                            visible: controller.alignLines.length > 0
                                && controller.canPreview

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

                                Item { Layout.preferredWidth: root.labelWidth }
                                Button {
                                    text: "Save as edited"
                                    enabled: controller.canRun
                                    onClicked: controller.saveEdited()
                                    HoverPanel {
                                        note: "Write the network as it is drawn "
                                            + "now into an 'edited' folder beside "
                                            + "the source, and read that from here on."
                                        showing: parent.hovered
                                    }
                                }
                                Item { Layout.fillWidth: true }
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

                        Item { Layout.fillHeight: true }
                    }
                }

                // what the run produces
                ColumnLayout {
                    SplitView.fillWidth: true
                    SplitView.minimumWidth: 470
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
                            Button { text: "Browse…"; onClicked: outputDialog.open() }
                            // Nothing is written into the chosen folder itself,
                            // so say where it does go before the run rather
                            // than in the status line afterwards.
                            InfoButton {
                                note: "Each run makes its own timestamped "
                                    + "folder in here, named for the mode and "
                                    + "the run. \"latest_result\" is kept "
                                    + "pointing at the newest."
                            }
                        }

                        Repeater {
                            model: controller.outputLines
                            delegate: FormLine {
                                required property var modelData
                                line: modelData
                            }
                        }

                        Rectangle {
                            Layout.fillWidth: true
                            Layout.topMargin: 2
                            Layout.leftMargin: -8
                            Layout.rightMargin: -8
                            implicitHeight: plate.implicitHeight + 20
                            color: root.plateColor
                            radius: 6

                            ColumnLayout {
                                id: plate
                                anchors.fill: parent
                                anchors.margins: 8
                                spacing: 8

                                Repeater {
                                    model: controller.formatLines
                                    delegate: FormLine {
                                        required property var modelData
                                        line: modelData
                                    }
                                }

                                RowLayout {
                                    Layout.fillWidth: true
                                    spacing: 8

                                    Item { Layout.preferredWidth: root.labelWidth }
                                    Button {
                                        text: "Select none"
                                        onClicked: controller.clearOutputs()
                                        HoverPanel {
                                            note: "Write no networks and no plots. "
                                                + "A run still leaves its report "
                                                + "and the batch a preview reads."
                                            showing: parent.hovered
                                        }
                                    }
                                    Button {
                                        text: "Defaults"
                                        onClicked: controller.resetOutputs()
                                        HoverPanel {
                                            note: "Networks as .csv, plots as .webp."
                                            showing: parent.hovered
                                        }
                                    }
                                    Item { Layout.fillWidth: true }
                                }
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

                            TabBar {
                                id: previewTabs
                                onCurrentIndexChanged:
                                    controller.selectPreviewTab(currentIndex)

                                // A mode with nothing synthetic to show must
                                // not be left sitting on that tab.
                                Connections {
                                    target: controller
                                    function onChanged() {
                                        if (!controller.offersSyntheticPreview
                                                && previewTabs.currentIndex !== 0)
                                            previewTabs.currentIndex = 0
                                    }
                                }

                                TabButton { text: "Original"; width: 112 }
                                TabButton {
                                    text: "Synthetic"
                                    width: 112
                                    // Absent outside generate: a hybrid network
                                    // is far too large to draw, and neither it
                                    // nor sweep writes the batch a preview reads.
                                    visible: controller.offersSyntheticPreview
                                    enabled: controller.canPreviewSynthetic
                                }
                            }

                            // Absent when the input came without an image: with
                            // nothing behind the network there is nothing to
                            // switch.
                            Button {
                                text: "Background"
                                checkable: true
                                checked: controller.showBackground
                                visible: previewTabs.currentIndex === 0
                                    && controller.hasBackground
                                onToggled: controller.setShowBackground(checked)
                            }
                            Button {
                                text: "Network"
                                checkable: true
                                checked: controller.showNetwork
                                visible: previewTabs.currentIndex === 0
                                onToggled: controller.setShowNetwork(checked)
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                text: controller.previewStatus
                                color: controller.previewFailed
                                    ? root.badColor : palette.text
                                opacity: controller.previewFailed ? 1.0 : 0.65
                                elide: Text.ElideRight
                            }
                        }

                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            opacity: 0.65
                            text: previewTabs.currentIndex === 0
                                ? controller.originalNote : controller.syntheticNote
                            visible: text !== ""
                        }

                        StackLayout {
                            Layout.fillWidth: true
                            Layout.preferredHeight: 264
                            currentIndex: previewTabs.currentIndex

                            PreviewPane {
                                image: controller.originalImage
                                info: controller.originalInfo
                                placeholder: controller.previewStatus !== ""
                                    ? controller.previewStatus
                                    : "Choose an edge list and positions, and the input network is drawn here."
                            }
                            PreviewPane {
                                image: controller.syntheticImage
                                info: controller.syntheticInfo
                                placeholder: controller.previewStatus !== ""
                                    ? controller.previewStatus
                                    : "Run, and the first network of the output is drawn here."
                            }
                        }
                    }

                    Card {
                        title: "Log"
                        Layout.leftMargin: 14
                        Layout.fillHeight: true
                        Layout.minimumHeight: 170

                        ScrollView {
                            Layout.fillWidth: true
                            Layout.fillHeight: true
                            TextArea {
                                id: logArea
                                readOnly: true
                                wrapMode: TextArea.NoWrap
                                font.family: "monospace"
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
