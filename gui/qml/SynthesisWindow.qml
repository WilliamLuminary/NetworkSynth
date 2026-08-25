import QtQuick
import QtQuick.Controls
import QtQuick.Dialogs
import QtQuick.Layouts

ApplicationWindow {
    id: root
    visible: true
    width: 1180
    height: 820
    minimumWidth: 900
    minimumHeight: 560
    title: "NetworkSynth — Synthesis"

    // `controller` is exposed from Python as a context property.
    //
    // Two panes: what a run is given on the left, what it produces on the
    // right.  Run and progress sit in the footer, so the button you press
    // every time is never scrolled off.

    readonly property int labelWidth: 140
    readonly property int numberWidth: 92

    // What a folded section is set to, so folding hides rows without hiding
    // state.  Re-read whenever `expanded` flips, which is the only moment it
    // can be read at all.
    function summaryOf(section) {
        var parts = []
        for (var i = 0; i < section.lines.length; i++) {
            var fields = section.lines[i].fields
            for (var j = 0; j < fields.length; j++) {
                var value = controller.valueOf(fields[j].id)
                if (fields[j].kind === "bool") {
                    if (value)
                        parts.push(fields[j].label)
                } else {
                    parts.push(value)
                }
            }
        }
        return parts.slice(0, 3).join("  ·  ")
    }

    header: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            Label {
                text: "Synthetic network generation"
                font.pixelSize: 16
                font.bold: true
                Layout.fillWidth: true
            }
            Label {
                text: "Mode"
                opacity: 0.6
            }
            ComboBox {
                id: modeBox
                model: controller.modeLabels
                currentIndex: controller.modes.indexOf(controller.mode)
                onActivated: controller.selectMode(currentIndex)
                enabled: !controller.running
                Layout.preferredWidth: 170
            }
        }
    }

    footer: ToolBar {
        RowLayout {
            anchors.fill: parent
            anchors.leftMargin: 12
            anchors.rightMargin: 12
            spacing: 12

            Button {
                text: controller.running ? "Cancel" : "Run"
                highlighted: !controller.running
                Layout.preferredWidth: 96
                onClicked: controller.running ? controller.cancel() : controller.run()
            }
            ProgressBar {
                // Only while there is progress to show: an empty trough beside
                // the button reads as something broken.
                visible: controller.running || controller.percent > 0
                Layout.preferredWidth: 220
                from: 0
                to: 100
                value: controller.percent
                indeterminate: controller.running && controller.percent <= 0
            }
            Label {
                text: controller.running ? controller.percent.toFixed(0) + "%" : ""
                Layout.preferredWidth: 40
            }
            Label {
                Layout.fillWidth: true
                elide: Text.ElideRight
                text: controller.status
                color: controller.failed ? "#c0392b" : palette.text
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

    // A section title that folds the rows under it away.
    component SectionHeader: ItemDelegate {
        property string name: ""
        property string summary: ""
        property bool open: true

        Layout.fillWidth: true
        contentItem: RowLayout {
            spacing: 6
            Label {
                text: (open ? "▾  " : "▸  ") + name.toUpperCase()
                font.bold: true
                font.pixelSize: 11
                opacity: 0.75
            }
            Label {
                Layout.fillWidth: true
                horizontalAlignment: Text.AlignRight
                elide: Text.ElideRight
                visible: !open
                text: summary
                opacity: 0.55
            }
        }
    }

    // One editor per field kind, as siblings gated on `visible` rather than a
    // Loader with inline Components: those get a context where neither
    // `controller` nor `modelData` resolves.  QtQuick.Layouts skips invisible
    // items, so only the relevant one takes space.
    component FieldEditor: RowLayout {
        required property var field
        required property var value
        signal edited(var newValue)
        signal sized(int index, var newValue)

        spacing: 6

        CheckBox {
            visible: field.kind === "bool"
            checked: value === true
            onToggled: edited(checked)
        }

        ComboBox {
            visible: field.kind === "choice"
            Layout.preferredWidth: 160
            model: field.options
            currentIndex: Math.max(0, field.options.indexOf(value))
            onActivated: edited(field.options[currentIndex])
        }

        TextField {
            visible: field.kind === "text"
            Layout.preferredWidth: 200
            text: field.kind === "text" ? value : ""
            onEditingFinished: edited(text)
        }

        TextField {
            visible: field.kind === "number" || field.kind === "integer"
            Layout.preferredWidth: root.numberWidth
            text: field.kind === "number" || field.kind === "integer" ? value : ""
            validator: DoubleValidator {}
            onEditingFinished: edited(text)
        }

        RowLayout {
            visible: field.kind === "size" || field.kind === "range"
            spacing: 6
            TextField {
                Layout.preferredWidth: root.numberWidth
                // Bindings of invisible siblings still evaluate, so guard the
                // indexing: on a scalar field this would be undefined.
                text: field.kind === "size" || field.kind === "range"
                    ? value[0] : ""
                validator: DoubleValidator { bottom: 0 }
                onEditingFinished: sized(0, text)
            }
            Label { text: field.kind === "range" ? "→" : "×"; opacity: 0.6 }
            TextField {
                Layout.preferredWidth: root.numberWidth
                text: field.kind === "size" || field.kind === "range"
                    ? value[1] : ""
                validator: DoubleValidator { bottom: 0 }
                onEditingFinished: sized(1, text)
            }
        }

        Item { Layout.fillWidth: true }
    }

    // One form row: a single field, or a set of switches sharing a label.
    component FormLine: RowLayout {
        required property var line

        Layout.fillWidth: true
        spacing: 8

        Label {
            text: line.row ? line.row : line.fields[0].label
            Layout.preferredWidth: root.labelWidth
            elide: Text.ElideRight
            ToolTip.text: line.row ? "" : line.fields[0].help
            ToolTip.visible: !line.row && line.fields[0].help ? lineHover.hovered : false
            HoverHandler { id: lineHover }
        }

        // A set: every switch on one line, as the choice it is.
        Repeater {
            model: line.row ? line.fields : []
            delegate: Button {
                required property var modelData
                text: modelData.label
                checkable: true
                checked: controller.valueOf(modelData.id) === true
                ToolTip.text: modelData.help
                ToolTip.visible: modelData.help ? hovered : false
                onToggled: controller.setValue(modelData.id, checked)
            }
        }

        FieldEditor {
            visible: !line.row
            Layout.fillWidth: true
            field: line.fields[0]
            value: controller.valueOf(line.fields[0].id)
            onEdited: (newValue) => controller.setValue(line.fields[0].id, newValue)
            onSized: (index, newValue) => controller.setSize(
                line.fields[0].id, index, newValue)
        }

        Item { Layout.fillWidth: true; visible: line.row }
    }

    // One preview pane, used by both sides.  Nothing in it refers to
    // `controller` — the two values are bound from outside, where that
    // certainly resolves.
    component PreviewPane: RowLayout {
        property alias image: picture.source
        property alias info: details.text

        spacing: 10

        Rectangle {
            Layout.preferredWidth: 300
            Layout.preferredHeight: 300
            color: "transparent"
            border.color: "#9e9e9e"
            border.width: 1

            Image {
                id: picture
                anchors.fill: parent
                anchors.margins: 1
                fillMode: Image.PreserveAspectFit
                asynchronous: true
                cache: false
            }
            Label {
                anchors.centerIn: parent
                visible: picture.source == ""
                text: "Nothing to draw"
                opacity: 0.5
            }
        }

        ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: 300
            TextArea {
                id: details
                readOnly: true
                wrapMode: TextArea.NoWrap
                font.family: "monospace"
                font.pixelSize: 11
            }
        }
    }

    SplitView {
        anchors.fill: parent
        orientation: Qt.Horizontal

        // ---- left: what the run is given ----
        ScrollView {
            SplitView.preferredWidth: 470
            SplitView.minimumWidth: 380
            contentWidth: availableWidth

            ColumnLayout {
                width: parent.width
                spacing: 10

                SectionHeader {
                    name: "Input"
                    open: true
                    onClicked: {}
                }

                ColumnLayout {
                    Layout.fillWidth: true
                    Layout.leftMargin: 8
                    spacing: 6

                    // Only where the mode offers a choice: analysis reads a
                    // results directory and nothing else.
                    RowLayout {
                        Layout.fillWidth: true
                        spacing: 8
                        visible: controller.inputShapes.length > 1

                        Label { text: "Format"; Layout.preferredWidth: root.labelWidth }
                        ComboBox {
                            id: shapeBox
                            Layout.fillWidth: true
                            model: controller.inputShapes
                            currentIndex: controller.inputShape
                            onActivated: controller.selectInputShape(currentIndex)
                            enabled: !controller.running
                        }
                    }

                    Repeater {
                        model: controller.inputs

                        delegate: RowLayout {
                            required property var modelData
                            Layout.fillWidth: true
                            spacing: 8

                            Label {
                                text: modelData.label
                                Layout.preferredWidth: root.labelWidth
                            }
                            TextField {
                                Layout.fillWidth: true
                                text: modelData.value
                                placeholderText: modelData.placeholder
                                onEditingFinished: controller.setInput(modelData.id, text)
                            }
                            Button {
                                text: "Browse…"
                                onClicked: root.browseFor(modelData)
                            }
                            // What this input's format has to contain.  Each
                            // shape reads a different one, and the wrong file
                            // fails deep inside a run rather than here.
                            Button {
                                id: contract
                                text: "!"
                                checkable: true
                                implicitWidth: 34
                                visible: modelData.help !== ""
                                ToolTip.text: modelData.help
                                ToolTip.visible: contract.checked
                                    || contractHover.hovered
                                HoverHandler { id: contractHover }
                            }
                        }
                    }
                }

                Repeater {
                    model: controller.configSections

                    delegate: ColumnLayout {
                        required property var modelData
                        readonly property var section: modelData
                        property bool expanded: !modelData.collapsed

                        Layout.fillWidth: true
                        spacing: 6

                        SectionHeader {
                            name: section.name
                            open: expanded
                            summary: expanded ? "" : root.summaryOf(section)
                            onClicked: expanded = !expanded
                        }

                        ColumnLayout {
                            Layout.fillWidth: true
                            Layout.leftMargin: 8
                            visible: expanded
                            spacing: 6

                            Repeater {
                                model: section.lines
                                delegate: FormLine {
                                    required property var modelData
                                    line: modelData
                                }
                            }
                        }
                    }
                }

                Item { Layout.fillHeight: true }
            }
        }

        // ---- right: what the run produces ----
        ColumnLayout {
            SplitView.fillWidth: true
            SplitView.minimumWidth: 420
            spacing: 10

            SectionHeader { name: "Output"; open: true; onClicked: {} }

            ColumnLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 8
                spacing: 6

                RowLayout {
                    Layout.fillWidth: true
                    spacing: 8
                    Label { text: "Folder"; Layout.preferredWidth: root.labelWidth }
                    TextField {
                        Layout.fillWidth: true
                        text: controller.outputDir
                        onEditingFinished: controller.setOutputDir(text)
                    }
                    Button { text: "Browse…"; onClicked: outputDialog.open() }
                }

                Repeater {
                    model: controller.outputLines
                    delegate: FormLine {
                        required property var modelData
                        line: modelData
                    }
                }
            }

            RowLayout {
                Layout.fillWidth: true
                Layout.topMargin: 4
                spacing: 8
                visible: controller.canPreview

                Label {
                    text: "PREVIEW"
                    font.bold: true
                    font.pixelSize: 11
                    opacity: 0.75
                }
                Button {
                    text: "Original"
                    checkable: true
                    checked: controller.previewOriginal
                    onToggled: controller.setPreviewOriginal(checked)
                }
                Button {
                    text: "Synthetic"
                    checkable: true
                    checked: controller.previewSynthetic
                    // Absent outside generate: a hybrid network is far too
                    // large to draw, and neither it nor sweep writes the batch
                    // a preview reads.
                    visible: controller.offersSyntheticPreview
                    enabled: controller.canPreviewSynthetic
                    onToggled: controller.setPreviewSynthetic(checked)
                }
                Label {
                    Layout.fillWidth: true
                    text: controller.previewStatus
                    color: controller.previewFailed ? "#c0392b" : palette.text
                    elide: Text.ElideRight
                }
            }

            ScrollView {
                Layout.fillWidth: true
                Layout.fillHeight: true
                visible: controller.previewOriginal || controller.previewSynthetic
                contentWidth: availableWidth

                ColumnLayout {
                    width: parent.width
                    spacing: 10

                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: controller.previewOriginal
                        spacing: 6

                        RowLayout {
                            Layout.fillWidth: true
                            spacing: 8

                            Label { text: "Original"; font.bold: true }
                            // Absent when the input came without an image:
                            // with nothing behind the network there is
                            // nothing to switch.
                            Button {
                                text: "Background"
                                checkable: true
                                checked: controller.showBackground
                                visible: controller.hasBackground
                                onToggled: controller.setShowBackground(checked)
                            }
                            Button {
                                text: "Network"
                                checkable: true
                                checked: controller.showNetwork
                                onToggled: controller.setShowNetwork(checked)
                            }
                            Label {
                                Layout.fillWidth: true
                                text: controller.originalNote
                                elide: Text.ElideRight
                                opacity: 0.7
                            }
                        }

                        PreviewPane {
                            Layout.fillWidth: true
                            image: controller.originalImage
                            info: controller.originalInfo
                        }
                    }

                    ColumnLayout {
                        Layout.fillWidth: true
                        visible: controller.previewSynthetic
                        spacing: 6

                        Label { text: "Synthetic"; font.bold: true }
                        Label {
                            Layout.fillWidth: true
                            wrapMode: Text.WordWrap
                            text: controller.syntheticNote
                            opacity: 0.7
                        }
                        PreviewPane {
                            Layout.fillWidth: true
                            image: controller.syntheticImage
                            info: controller.syntheticInfo
                        }
                    }
                }
            }

            Item {
                Layout.fillHeight: true
                visible: !(controller.previewOriginal || controller.previewSynthetic)
            }

            GroupBox {
                title: "Log"
                Layout.fillWidth: true
                Layout.preferredHeight: 190

                ScrollView {
                    anchors.fill: parent
                    TextArea {
                        id: logArea
                        readOnly: true
                        wrapMode: TextArea.NoWrap
                        font.family: "monospace"
                        font.pixelSize: 11
                        text: controller.logText
                        onTextChanged: cursorPosition = length
                    }
                }
            }
        }
    }
}
