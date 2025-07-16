import datetime
import inspect

from sqlalchemy import BigInteger, Boolean, Column, Date, ForeignKey, Integer, Numeric, String, UniqueConstraint, Text
from sqlalchemy.orm import relationship, validates, DeclarativeMeta

from app.core.database import Base


class WheelSet(Base):
    __tablename__ = "wheel_set"
    __table_args__ = (
        UniqueConstraint("date_detect", "part_number", "storage_name"),
        {"comment": "Браковка колесных пар"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    load_from = Column(String(10), comment="Источник данных")
    date_upload = Column(Date, default=datetime.date.today, comment="Дата загрузки")
    wagon_num = Column(BigInteger, index=True, comment="Номер вагона")
    branch_name = Column(String(40), comment="Филиал")
    railway = Column(String(40), comment="Дорога")
    date_detect = Column(Date, nullable=False, comment="Дата выявления хищения")
    part_number = Column(
        String(40), index=True, nullable=False, comment="Номер похищенной детали (Завод-номер-детали-год)"
    )
    thickness_left_rim = Column(Integer, comment="Толщина левого обода, мм")
    thickness_right_rim = Column(Integer, comment="Толщина правого обода, мм")
    rejection_code = Column(Text, comment="Код браковки (причина)")
    part_cost = Column(Numeric(precision=10, scale=2), comment="Стоимость похищенной детали, руб.")
    mounting_type = Column(String(20), comment="Описание типа крепления")
    system = Column(String(20), comment="Описание системы")
    storage_name = Column(String(40), nullable=False, comment="Обозначение склада")
    external_contract_num = Column(String(40), comment="Внешний номер договора")
    document_num = Column(String(20), comment="Документ")
    #   Далее поля, идентичные модели Steal
    repay = Column(Numeric(precision=10, scale=2), comment="Всего возмещено, руб.")
    repay_author = Column(Numeric(precision=10, scale=2), comment="Восстановление вагона за счёт ВРП/виновником, руб.")
    insurance_notification_date = Column(Date, comment="Дата направления уведомления в СК")
    insurance_notification_num = Column(String(40), comment="Исходящий номер уведомления в СК")
    insurance_name = Column(String(40), comment="Наименование СК")
    is_insurance_of_carrier = Column(Boolean, default=True, comment="Страховая компания перевозчика (да/нет)")
    insurance_number = Column(String(40), comment="Номер убытка (присваивается СК)")
    insurance_claim_number = Column(String(40), comment="Номер претензии")
    insurance_payment_total = Column(
        Numeric(precision=10, scale=2), comment="Общий размер требований о выплате СК, руб."
    )
    insurance_payment_date = Column(Date, comment="Дата выплаты страхового возмещения")
    insurance_payment_done = Column(Numeric(precision=10, scale=2), comment="Выплаченная сумма СК, руб.")
    author_pretension_date = Column(Date, comment="Дата направления претензии виновному")
    author_name = Column(String(40), comment="Наименование виновника")
    author_pretension_number = Column(String(40), comment="Исходящий номер претензии")
    author_payment_total = Column(Numeric(precision=10, scale=2), comment="Общий размер требований о выплате, руб.")
    author_payment_date = Column(Date, comment="Дата выплаты виновником по претензии")
    author_payment_done = Column(Numeric(precision=10, scale=2), comment="Выплаченная сумма, руб.")
    author_lawyer_date = Column(Date, comment="Дата передачи материала Юристам")
    author_lawyer_number = Column(String(40), comment="Номер Арбитражного дела")
    police_date = Column(Date, comment="Дата передачи материала БР")
    police_ovd_date = Column(Date, comment="Дата направления заявления в ОВД")
    police_ovd_name = Column(String(40), comment="Наименование ОВД")
    police_payment = Column(Numeric(precision=10, scale=2), comment="Размер ущерба, руб.")
    police_decision = Column(
        String(40), comment="Процессуальное решение по заявлению (возбуждение уг. дела / отказ в ВУД)"
    )
    police_decision_date = Column(Date, comment="Дата процессуального решения")
    # register of damaged parts
    # 1
    claim_refund_partner_1 = Column(String(40), comment="Наименование Контрагента по возмещению 1")
    claim_date_1 = Column(Date, comment="Дата претензии 1")
    claim_internal_number_1 = Column(String(25), comment="Внутренний № претензии 1")
    claim_external_number_1 = Column(String(25), comment="Внешний № претензии 1")
    claim_sum_1 = Column(Numeric(precision=10, scale=2), comment="Сумма претензии 1, руб.")
    claim_refund_date_1 = Column(Date, comment="Дата возмещения 1")
    claim_sum_damage_1 = Column(Numeric(precision=10, scale=2), comment="Полученная сумма 1, руб.")
    claim_partner_payed_1 = Column(Numeric(precision=10, scale=2), comment="Возмещено страховой компанией 1, руб.")
    # 2
    claim_refund_partner_2 = Column(String(40), comment="Наименование Контрагента по возмещению 2")
    claim_date_2 = Column(Date, comment="Дата претензии 2")
    claim_internal_number_2 = Column(String(25), comment="Внутренний № претензии 2")
    claim_external_number_2 = Column(String(25), comment="Внешний № претензии 2")
    claim_sum_2 = Column(Numeric(precision=10, scale=2), comment="Сумма претензии 2, руб.")
    claim_refund_date_2 = Column(Date, comment="Дата возмещения 2")
    claim_sum_damage_2 = Column(Numeric(precision=10, scale=2), comment="Полученная сумма 2, руб.")
    claim_partner_payed_2 = Column(Numeric(precision=10, scale=2), comment="Возмещено страховой компанией 2, руб.")
    # 3
    claim_refund_partner_3 = Column(String(40), comment="Наименование Контрагента по возмещению 3")
    claim_date_3 = Column(Date, comment="Дата претензии 3")
    claim_internal_number_3 = Column(String(25), comment="Внутренний № претензии 3")
    claim_external_number_3 = Column(String(25), comment="Внешний № претензии 3")
    claim_sum_3 = Column(Numeric(precision=10, scale=2), comment="Сумма претензии 3, руб.")
    claim_refund_date_3 = Column(Date, comment="Дата возмещения 3")
    claim_sum_damage_3 = Column(Numeric(precision=10, scale=2), comment="Полученная сумма 3, руб.")
    claim_partner_payed_3 = Column(Numeric(precision=10, scale=2), comment="Возмещено страховой компанией 3, руб.")
    # sum
    claim_sum_all = Column(Numeric(precision=10, scale=2), comment="Сумма претензии, итого, руб.")
    claim_sum_damage_all = Column(Numeric(precision=10, scale=2), comment="Полученная сумма ущерба, итого, руб.")
    claim_partner_payed_all = Column(
        Numeric(precision=10, scale=2), comment="Возмещено страховой компанией, итого, руб."
    )

    @validates("part_cost")
    def validate_part_cost(self, key, val):
        return val if val else 0


class WheelSetFilter(Base):
    __tablename__ = "wheel_set_filter"
    __table_args__ = {
        "comment": "Справочник причин браковки для анализа колесных пар",
    }

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    name = Column(String(160), unique=True, comment="Наименование")


class MountingType(Base):
    __tablename__ = "mounting_type"
    __table_args__ = {"comment": "Справочник типов креплений"}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    name = Column(String(20), nullable=False, unique=True, comment="Наименование")
    is_loading = Column(Boolean, default=False, comment="Загружаем (да/нет)")

    @validates("name")
    def convert_upper(self, key, value):
        return value.upper()


class MountingTypeMap(Base):
    __tablename__ = "mounting_type_map"
    __table_args__ = {"comment": "Маппинг типов крепления для привязки к справочнику цен (WheelSetCost)"}

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    name_from = Column(String(20), nullable=False, unique=True, comment="Тип крепления в формах загрузки")
    name_to = Column(String(20), nullable=False, comment="Тип крепления в справочнике цен")

    @validates("name_from", "name_to")
    def convert_upper(self, key, value):
        return value.upper()


class WheelSetCost(Base):
    __tablename__ = "wheel_set_cost"
    __table_args__ = (
        UniqueConstraint(
            "rim_thickness_min", "rim_thickness_max", "mounting_type", "is_same_cost_for_all_branch", "branch_name"
        ),
        {"comment": "Справочник цен колесных пар"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    rim_thickness_min = Column(Integer, nullable=False, comment="Толщина обода мин")
    rim_thickness_max = Column(Integer, nullable=False, comment="Толщина обода макс")
    mounting_type = Column(String(20), nullable=False, comment="Тип крепления")
    branch_name = Column(String(40), nullable=False, comment="Филиал")
    is_same_cost_for_all_branch = Column(Boolean, comment="Цена для всех филиалов одинаковая (да/нет)")
    cost = Column(Numeric(precision=10, scale=2), comment="Цена без НДС, руб.")
    name = Column(String(60), comment="Наименование деталей")

    @validates("mounting_type")
    def convert_upper(self, key, value):
        return value.upper()


class Storage(Base):
    __tablename__ = "storage"
    __table_args__ = {
        "comment": "Справочник складов",
    }

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    # load_from = Column(String(10), comment="Загружено из SAP/АСУ ВРК/Варекс")
    name = Column(String(40), unique=True, comment="Обозначение склада/Депо")
    branch = Column(String(40), comment="Филиал")
    railway = Column(String(40), comment="Дорога")


# Models for "Damage Compensation"
class WorkFilter(Base):
    __tablename__ = "work_filter"
    __table_args__ = {
        "comment": "Справочник видов работ анализируемых при возмещении убытков",
    }

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    name = Column(String(160), unique=True, comment="Наименование")

    storage_list = relationship("WorkFilterByStorage", cascade="all,delete", backref="work_filter")


class WorkFilterByStorage(Base):
    __tablename__ = "work_filter_by_storage"
    __table_args__ = {
        "comment": "Уточнение справочника вида работ в разрезе складов филиалов",
    }

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    work_filter_id = Column(ForeignKey("work_filter.id"), comment="ИД в справочнике вида работ")
    branch_name = Column(String(40), comment="Филиал")
    storage_name = Column(String(40), comment="Склад")
    number_rdv = Column(String(2), nullable=True, comment="Номер позиции РДВ")
    work_name_by_document = Column(
        String(40), nullable=True, comment="Наименование вида работ в рамках документа (несколько строк)"
    )


class WorkCost(Base):
    __tablename__ = "work_cost"
    __table_args__ = (
        UniqueConstraint("name", "number", "cost"),
        {"comment": "Справочник видов работ для заполнения пустых строк в файле выгрузки из SAP"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    name = Column(String(160), comment="Наименование")
    number = Column(String(40), comment="Номер договора ПГК")
    cost = Column(Numeric(precision=10, scale=2), comment="Цена (руб.)")


class Steal(Base):
    __tablename__ = "steal"
    __table_args__ = (
        UniqueConstraint("document_num", "document_num_num"),
        {"comment": "Разоборудование (кражи) вагонов"},
    )

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    date_upload = Column(Date, default=datetime.date.today, comment="Дата загрузки")
    document_num = Column(BigInteger, index=True, comment="Документ (ИД)")
    document_num_num = Column(Integer, index=True, comment="Номер услуги (в рамках документа)")
    period_load = Column(String(7), comment="Период загрузки")
    # document_date = Column(Date, comment="Дата проводки документа")
    date_detect = Column(Date, comment="Дата выявления разоборудования (ремонта вагона)")
    branch_name = Column(String(40), comment="Филиал")
    storage_name = Column(String(40), comment="Склад")
    wagon_num = Column(BigInteger, index=True, comment="№ вагона")
    external_contract_num = Column(String(40), comment="Внешний номер договора")
    part_name = Column(String(160), comment="Наименование похищенной детали (номенклатура)")
    part_cost = Column(Numeric(precision=10, scale=2), comment="Стоимость похищенных деталей (руб.)")
    part_amount = Column(Integer, comment="Количество")
    repay = Column(Numeric(precision=10, scale=2), comment="Всего возмещено (руб.)")
    repay_author = Column(Numeric(precision=10, scale=2), comment="Восстановление вагона за счёт ВРП/виновником (руб.)")
    insurance_notification_date = Column(Date, comment="Дата направления уведомления в СК")
    insurance_notification_num = Column(String(40), comment="Исходящий номер уведомления в СК")
    insurance_name = Column(String(40), comment="Наименование СК")
    is_insurance_of_carrier = Column(Boolean, default=True, comment="Страховая компания перевозчика (да/нет)")
    insurance_number = Column(String(40), comment="Номер убытка (присваивается СК)")
    insurance_claim_number = Column(String(40), comment="Номер претензии")
    insurance_payment_total = Column(Numeric(precision=10, scale=2), comment="Общий размер требований о выплате")
    insurance_payment_date = Column(Date, comment="Дата выплаты страхового возмещения")
    insurance_payment_done = Column(Numeric(precision=10, scale=2), comment="Выплаченная сумма (руб.)")
    author_pretension_date = Column(Date, comment="Дата направления претензии виновному")
    author_name = Column(String(40), comment="Наименование виновника")
    author_pretension_number = Column(String(40), comment="Исходящий номер претензии")
    author_payment_total = Column(Numeric(precision=10, scale=2), comment="Общий размер требований о выплате")
    author_payment_date = Column(Date, comment="Дата выплаты виновником по претензии")
    author_payment_done = Column(Numeric(precision=10, scale=2), comment="Выплаченная сумма (руб.)")
    author_lawyer_date = Column(Date, comment="Дата передачи материала Юристам")
    author_lawyer_number = Column(String(40), comment="Номер Арбитражного дела")
    police_date = Column(Date, comment="Дата передачи материала БР")
    police_ovd_date = Column(Date, comment="Дата направления заявления в ОВД")
    police_ovd_name = Column(String(40), comment="Наименование ОВД")
    police_payment = Column(Numeric(precision=10, scale=2), comment="Размер ущерба (руб.)")
    police_decision = Column(
        String(40), comment="Процессуальное решение по заявлению (возбуждение уг. дела / отказ в ВУД)"
    )
    police_decision_date = Column(Date, comment="Дата процессуального решения")


class CheckFilter(Base):
    __tablename__ = "check_filter"
    __table_args__ = {
        "comment": "Результат проверки соответствия фильтра работ 'work_filter' и вх. Excel-файла",
    }

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment="ID")
    filter_name = Column(String(160), comment="Наименование в файле 'work_filter'")
    work_name = Column(String(160), comment="Наименование во вх. Excel-файле")
    distance = Column(Integer, comment="дистанция/количество перестановок между наименованиями")
    count = Column(Integer, comment="количество")