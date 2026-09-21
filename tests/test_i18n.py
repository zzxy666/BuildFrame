from PyQt5 import QtCore

from labelCloud.view.i18n import language_manager, normalize_language, tr


def _application():
    return QtCore.QCoreApplication.instance() or QtCore.QCoreApplication([])


def test_language_switch_translates_static_and_formatted_text():
    app = _application()
    try:
        language_manager.set_language("en_US")
        assert tr("点云滤波") == "Filter Point Cloud"
        assert (
            tr("当前：<em>{name}</em>", name="sample.ply")
            == "Current: <em>sample.ply</em>"
        )

        language_manager.set_language("zh_CN")
        assert tr("点云滤波") == "点云滤波"
    finally:
        language_manager.set_language("zh_CN")
        app.processEvents()


def test_unknown_language_falls_back_to_chinese():
    assert normalize_language("unsupported") == "zh_CN"
