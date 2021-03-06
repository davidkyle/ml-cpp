/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License;
 * you may not use this file except in compliance with the Elastic License.
 */
#include "CRuleConditionTest.h"

#include <core/CLogger.h>

#include <model/CAnomalyDetectorModel.h>
#include <model/CDataGatherer.h>
#include <model/CDetectionRule.h>
#include <model/CModelParams.h>
#include <model/CRuleCondition.h>
#include <model/CSearchKey.h>
#include <model/ModelTypes.h>

#include "Mocks.h"

#include <string>
#include <vector>

using namespace ml;
using namespace model;

namespace {

using TStrVec = std::vector<std::string>;

const std::string EMPTY_STRING;
}

CppUnit::Test* CRuleConditionTest::suite() {
    CppUnit::TestSuite* suiteOfTests = new CppUnit::TestSuite("CRuleConditionTest");

    suiteOfTests->addTest(new CppUnit::TestCaller<CRuleConditionTest>(
        "CRuleConditionTest::testTimeContition", &CRuleConditionTest::testTimeContition));

    return suiteOfTests;
}

void CRuleConditionTest::testTimeContition() {
    core_t::TTime bucketLength = 100;
    core_t::TTime startTime = 100;
    CSearchKey key;
    SModelParams params(bucketLength);
    CAnomalyDetectorModel::TFeatureInfluenceCalculatorCPtrPrVecVec influenceCalculators;

    model_t::TFeatureVec features;
    features.push_back(model_t::E_IndividualMeanByPerson);
    CAnomalyDetectorModel::TDataGathererPtr gathererPtr(new CDataGatherer(
        model_t::E_Metric, model_t::E_None, params, EMPTY_STRING, EMPTY_STRING,
        EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, EMPTY_STRING, TStrVec(),
        false, key, features, startTime, 0));

    CMockModel model(params, gathererPtr, influenceCalculators);

    {
        CRuleCondition condition;
        condition.type(CRuleCondition::E_Time);
        condition.condition().s_Op = CRuleCondition::E_GTE;
        condition.condition().s_Threshold = 500;

        CPPUNIT_ASSERT(condition.isNumerical());
        CPPUNIT_ASSERT(condition.isCategorical() == false);

        model_t::CResultType resultType(model_t::CResultType::E_Final);
        CPPUNIT_ASSERT(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                      resultType, false, std::size_t(0),
                                      std::size_t(1), core_t::TTime(450)) == false);
        CPPUNIT_ASSERT(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                      resultType, false, std::size_t(0),
                                      std::size_t(1), core_t::TTime(550)));
    }

    {
        CRuleCondition condition;
        condition.type(CRuleCondition::E_Time);
        condition.condition().s_Op = CRuleCondition::E_LT;
        condition.condition().s_Threshold = 600;

        CPPUNIT_ASSERT(condition.isNumerical());
        CPPUNIT_ASSERT(condition.isCategorical() == false);

        model_t::CResultType resultType(model_t::CResultType::E_Final);
        CPPUNIT_ASSERT(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                      resultType, false, std::size_t(0),
                                      std::size_t(1), core_t::TTime(600)) == false);
        CPPUNIT_ASSERT(condition.test(model, model_t::E_IndividualCountByBucketAndPerson,
                                      resultType, false, std::size_t(0),
                                      std::size_t(1), core_t::TTime(599)));
    }
}
